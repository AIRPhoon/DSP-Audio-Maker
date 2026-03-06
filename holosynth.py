"""
PyHoloSynth - 3D 物理与数学音频合成引擎 (v1.0)

这是一个无需依赖任何外部臃肿音频软件，纯基于 NumPy 和 SciPy 的高级音频合成库。
它通过极简的“二维参数矩阵”，即可实现母带级的多普勒空间平移、FM频率调制、
减法共振滤波与全息卷积混响。

【核心概念：乐谱矩阵 (Melody Matrix)】
每一条音轨(Track)都是一个列表，列表中的每一行代表一个声音事件，必须遵循以下 6 维格式：
[音符, 八度, 节拍数, 空间位置, 包络线, 音色参数]

1. 音符(Notes): 
    - 单音: 0
    - 休止符(静音): -1
    - 跨八度滑音: (0, 12) 
    - 和弦: [0, 2, 4] 
    - 和弦滑动: [(0, 7), (2, 9)]
2. 八度(Octave): 整数。0为基准，1为高八度，-1为低八度。
3. 节拍(Beats): 浮点数。声音持续的相对长度 (受全局 bpm 控制)。
4. 空间(Position): [角度, 距离]。
    - 角度: 0正前, 90极右, 180正后, 270极左。
    - 距离: 0贴脸，越大越远。
    - 动态多普勒滑行: [(270, 90), (5, 0)] (支持元组起止点)
5. 包络(Envelope): [[时间%, 音量%], ...]。建议使用 Presets.Envelopes。
6. 音色(Synth): [振幅, FM频率比, FM深度(可渐变), 截止频率, Q值, 噪声]。建议使用 Presets.Instruments。

【快速入门】
>>> from holosynth import Synthesizer, Presets
>>> synth = Synthesizer(bpm=120)
>>> track = [[0, 0, 1.0, [0, 1], Presets.Envelopes.PLUCK, Presets.Instruments.DX7_BELL]]
>>> mix_audio = synth.mix([track])
>>> synth.play(mix_audio)
"""

import sounddevice as sd
import numpy as np
import scipy.signal as signal

# ========================================================
# 预设库 (Presets)
# ========================================================
class Presets:
    """
    预设字典。包含常用的音色(Instruments)、包络线(Envelopes)和调式音阶(Scales)。
    你可以直接在矩阵中调用它们，例如：Presets.Instruments.DX7_BELL
    """
    class Instruments:
        """格式: [振幅, FM频率比, FM深度(可渐变), 截止频率, Q值, 噪声强度]"""
        SINE_PURE   = [0.8, 1.0, 0.0, 20000, 0.7, 0.0]        
        DX7_BELL    = [0.8, 3.5, (6.0, 0.0), 20000, 0.7, 0.0] 
        ACID_BASS   = [0.6, 1.0, 5.0, 800, 5.0, 0.0]          
        WARM_PAD    = [0.5, 1.01, 1.5, 2500, 1.0, 0.02]       
        PLUCK_BASS  = [0.7, 1.0, (3.0, 0.0), 1200, 2.0, 0.0]  
        DARK_BRASS  = [0.6, 1.0, (0.0, 3.0), 1500, 1.5, 0.0]  
        GLASS_PAD   = [0.4, 2.01, 2.0, 6000, 0.5, 0.0]        
        CHIPTUNE    = [0.5, 1.0, 15.0, 20000, 0.1, 0.0]       
        LOFI_KEYS   = [0.7, 1.0, 0.5, 1000, 0.8, 0.08]        
        CYBER_SNARE = [0.8, 0.5, 0.0, 10000, 1.0, 3.0]        
        NOISE_WIND  = [0.4, 1.0, 0.0, 3000, 1.5, 1.0]         

    class Envelopes:
        """格式: [[时间比例, 音量比例], ...]"""
        PLUCK   = [[0.0, 0], [0.05, 1], [0.3, 0]]       
        STACC   = [[0.0, 0], [0.1, 1], [0.8, 1], [1,0]] 
        LEGATO  = [[0.0, 0], [0.1, 1], [1.0, 1]]        
        PAD     = [[0.0, 0], [0.5, 1], [1.0, 0]]        
        SWELL   = [[0.0, 0], [0.8, 1], [1.0, 0]]        

    class Scales:
        """定义一个八度内的音程间隔 (半音数)"""
        MAJOR          = [0, 2, 4, 5, 7, 9, 11]  
        MINOR          = [0, 2, 3, 5, 7, 8, 10]  
        PENTATONIC     = [0, 2, 4, 7, 9]         
        BLUES          = [0, 3, 5, 6, 7, 10]     
        HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]  
        DORIAN         = [0, 2, 3, 5, 7, 9, 10]  
        WHOLE_TONE     = [0, 2, 4, 6, 8, 10]     
        HIRAJOSHI      = [0, 2, 3, 7, 8]         

# ========================================================
# 底层 DSP 工具类 (内部使用)
# ========================================================
class _DSPCore:
    """内部数字信号处理核心。包含插值包络、双二阶滤波器和多普勒空间渲染器。"""
    @staticmethod
    def envelope(num_samples, control_points):
        t_axis = np.linspace(0, 1.0, num_samples)
        pts = sorted(control_points, key=lambda x: x[0])
        if pts[0][0] > 0.0: pts.insert(0, [0.0, 0.0])
        if pts[-1][0] < 1.0: pts.append([1.0, 0.0])
        xp, fp = zip(*pts)
        return np.interp(t_axis, xp, fp)

    @staticmethod
    def biquad_lpf(wave, cutoff_hz, q_factor, fs):
        cutoff = np.clip(cutoff_hz, 20.0, fs / 2.1)
        q = np.clip(q_factor, 0.1, 20.0) 
        w0 = 2 * np.pi * cutoff / fs
        alpha = np.sin(w0) / (2 * q)
        b = np.array([(1-np.cos(w0))/2, 1-np.cos(w0), (1-np.cos(w0))/2]) / (1+alpha)
        a = np.array([1, -2*np.cos(w0)/(1+alpha), (1-alpha)/(1+alpha)])
        return signal.lfilter(b, a, wave)

    @staticmethod
    def spatial_panning(mono_wave, angle_array, dist_array):
        num_samples = len(mono_wave)
        t_indices = np.arange(num_samples) 
        dist_gain = 1.0 / (1.0 + np.clip(dist_array, 0, None))
        x_proj = np.sin(np.radians(angle_array))
        
        pan_p = (x_proj + 1.0) / 2.0 
        gain_l, gain_r = np.cos(pan_p * (np.pi/2)), np.sin(pan_p * (np.pi/2))
        
        delay_l = 26.0 * np.clip(x_proj, 0, None)
        delay_r = 26.0 * np.clip(-x_proj, 0, None)
        
        wave_l = np.interp(t_indices - delay_l, t_indices, mono_wave) * gain_l * dist_gain
        wave_r = np.interp(t_indices - delay_r, t_indices, mono_wave) * gain_r * dist_gain
        return np.column_stack((wave_l, wave_r))

# ========================================================
# 核心合成器 API
# ========================================================
class Synthesizer:
    """
    PyHoloSynth 的核心引擎控制台。负责矩阵解析、轨道渲染、混音与全局特效。
    
    常见报错排雷:
    1. ValueError: too many values to unpack
       原因: 和弦级滑音包裹了太多层中括号。正确写法是 [[(0, 2), (2, 4)], ...]
    2. 播放时有爆音/劈啪声
       原因: 休止符前的最后一个音符包络没有归零。请确保其包络线结尾为 [1.0, 0.0]。
    """

    def __init__(self, sample_rate=44100, bpm=120, root_offset=0, scale_intervals=Presets.Scales.MAJOR, ref_a=440.0):
        """
        初始化全息合成器实例。
        
        :param sample_rate: 音频采样率 (默认 44100Hz)。
        :param bpm: 歌曲的全局速度 (Beats Per Minute)。
        :param root_offset: 转调参数。0为C调，1为C#调，-3为A调。
        :param scale_intervals: 调式音程数组。引擎自带无限八度折叠算法。
        :param ref_a: 标准音高 (默认 A4=440.0Hz)。
        """
        self.sample_rate = sample_rate
        self.bpm = bpm
        self.root_offset = root_offset
        self.scale_intervals = scale_intervals
        self.ref_a = ref_a
        self._update_freqs()

    def _update_freqs(self):
        c4_freq = self.ref_a * (2 ** (-9 / 12)) * (2 ** (self.root_offset / 12))
        self.base_freqs = [c4_freq * (2 ** (iv / 12)) for iv in self.scale_intervals]

    def set_bpm(self, bpm):
        """动态修改全局 BPM 速度"""
        self.bpm = bpm

    def set_scale(self, root_offset, scale_intervals):
        """动态转调或更改音阶"""
        self.root_offset = root_offset
        self.scale_intervals = scale_intervals
        self._update_freqs()

    def _synth_voice(self, f_array, duration_s, init_phase, global_t, params):
        amp, fm_ratio, fm_idx, cutoff, q, noise = (list(params) + Presets.Instruments.SINE_PURE)[:6]
        num_samples = len(f_array)
        fm_idx_s, fm_idx_e = fm_idx if isinstance(fm_idx, (tuple, list)) else (fm_idx, fm_idx)
        fm_idx_array = np.linspace(fm_idx_s, fm_idx_e, num_samples)
        
        carrier_phases = init_phase + np.cumsum(2 * np.pi * f_array / self.sample_rate)
        mod_phases = np.cumsum(2 * np.pi * (f_array * fm_ratio) / self.sample_rate)
        
        wave = amp * np.sin(carrier_phases + fm_idx_array * np.sin(mod_phases))
        if noise > 0: wave += np.random.normal(0, noise, num_samples)
        if cutoff < 20000: wave = _DSPCore.biquad_lpf(wave, cutoff, q, self.sample_rate)
            
        return wave, carrier_phases[-1]

    def render_track(self, melody_matrix):
        """
        将单条乐谱矩阵渲染为 3D 立体声音频数组。
        
        :param melody_matrix: 包含音符与参数事件的二维列表。
        :return: np.int16 格式的双声道立体声 NumPy 数组。
        """
        audio_segments, voice_phases, global_t = [], {}, 0.0     
        sec_per_beat = 60.0 / self.bpm
        scale_len = len(self.base_freqs)
        
        for row in melody_matrix:
            row_data = list(row) + [None] * (6 - len(row))
            notes, octave, beats, position, vol_pts, synth_params = row_data[:6]
            
            if notes is None: continue
            notes = [notes] if isinstance(notes, (int, float, tuple)) else notes
            octave, beats = int(octave or 0), float(beats or 1.0)
            
            pos = position or [0, 0]
            a_p, d_p = pos[0], pos[1]
            a_start, a_end = a_p if isinstance(a_p, (tuple, list)) else (a_p, a_p)
            d_start, d_end = d_p if isinstance(d_p, (tuple, list)) else (d_p, d_p)
            
            duration_s = beats * sec_per_beat
            num_samples = int(self.sample_rate * duration_s)
            
            angle_array = np.linspace(a_start, a_end, num_samples)
            dist_array = np.linspace(d_start, d_end, num_samples)
            
            vol_pts = vol_pts or Presets.Envelopes.PAD
            synth_params = synth_params or Presets.Instruments.SINE_PURE
            
            mixed_stereo_wave = np.zeros((num_samples, 2))
            
            if notes == [-1]:
                audio_segments.append(mixed_stereo_wave) 
                global_t, voice_phases = global_t + duration_s, {}
                continue
                
            for v_idx, n_item in enumerate(notes):
                n_start, n_end = n_item if isinstance(n_item, (tuple, list)) else (n_item, n_item)
                
                idx_s, oct_s = int(n_start) % scale_len, int(n_start) // scale_len
                idx_e, oct_e = int(n_end) % scale_len, int(n_end) // scale_len
                
                f_s = self.base_freqs[idx_s] * (2 ** (octave + oct_s))
                f_e = self.base_freqs[idx_e] * (2 ** (octave + oct_e))
                f_array = np.linspace(f_s, f_e, num_samples)
                
                mono_wave, f_phase = self._synth_voice(f_array, duration_s, voice_phases.get(v_idx, 0.0), global_t, synth_params)
                voice_phases[v_idx] = f_phase  
                
                mixed_stereo_wave += _DSPCore.spatial_panning(mono_wave, angle_array, dist_array)
                
            env = _DSPCore.envelope(num_samples, vol_pts)
            audio_segments.append(mixed_stereo_wave * env[:, np.newaxis])
            global_t += duration_s
        
        if not audio_segments: return np.zeros((0, 2), dtype=np.int16)
        final_audio = np.concatenate(audio_segments)
        max_amp = np.max(np.abs(final_audio))
        return np.int16(final_audio / max_amp * 32767) if max_amp > 0 else np.int16(final_audio)

    def mix(self, tracks_list):
        """
        多轨混音台。并行渲染多条音轨，自动对齐时间轴并进行防爆音物理叠加。
        
        :param tracks_list: 包含多个乐谱矩阵的列表，例如 [track_bass, track_lead]。
        :return: 混合后的母带立体声数组。
        """
        rendered = [self.render_track(trk) for trk in tracks_list]
        max_len = max([len(trk) for trk in rendered]) if rendered else 0
        master_mix = np.zeros((max_len, 2), dtype=np.float64)
        
        for trk in rendered:
            padded = np.pad(trk, ((0, max_len - len(trk)), (0, 0)), mode='constant')
            master_mix += padded.astype(np.float64) / 32767.0
            
        max_amp = np.max(np.abs(master_mix))
        if max_amp > 0: master_mix /= max_amp
        return np.int16(master_mix * 32767)

    def apply_reverb(self, audio_int16, room_size=2.5, damping=0.6, wet_level=0.3):
        """
        为全局母带挂载 3D 卷积混响 (Convolution Reverb) 空间特效。
        
        :param audio_int16: 由 mix() 或 render_track() 生成的干声音频数组。
        :param room_size: 房间大小(秒)。决定混响尾音的消散时长。2.0为教堂，4.0为星际机库。
        :param damping: 高频阻尼(0.0~1.0)。空气与墙壁吸收的高频比例。越大声音越"闷"。
        :param wet_level: 干湿比(0.0~1.0)。0为纯干声，1.0为纯环境反射声。
        :return: 带有混响包裹感的立体声数组。
        """
        audio_float = audio_int16.astype(np.float64) / 32767.0
        length = int(self.sample_rate * room_size)
        
        ir_l, ir_r = np.random.normal(0, 1, length), np.random.normal(0, 1, length)
        env = np.exp(-5.0 * np.arange(length) / length)
        ir_l, ir_r = signal.lfilter([1-damping], [1, -damping], ir_l*env), signal.lfilter([1-damping], [1, -damping], ir_r*env)

        wet_l, wet_r = signal.fftconvolve(audio_float[:, 0], ir_l), signal.fftconvolve(audio_float[:, 1], ir_r)
        
        pad_len = len(wet_l) - len(audio_float)
        dry_l, dry_r = np.pad(audio_float[:,0], (0, pad_len)), np.pad(audio_float[:,1], (0, pad_len))

        out_l, out_r = dry_l*(1-wet_level) + wet_l*wet_level, dry_r*(1-wet_level) + wet_r*wet_level
        out = np.column_stack((out_l, out_r))
        
        max_amp = np.max(np.abs(out))
        return np.int16(out / max_amp * 32767) if max_amp > 0 else np.int16(out)

    def play(self, audio_data):
        """
        调用系统声卡实时播放生成的音频。
        :param audio_data: 待播放的音频数组。
        """
        sd.play(audio_data, self.sample_rate)
        sd.wait()