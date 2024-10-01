import numpy as np
from scipy.io import wavfile


class SoundWaveFactory:
    SAMPLING_RATE = 44100  # like 44.1 KHz
    DURATION_SECONDS = 5
    SOUND_ARRAY_LEN = SAMPLING_RATE * DURATION_SECONDS
    MAX_AMPLITUDE = 2 ** 13

    # From a list of https://en.wikipedia.org/wiki/Piano_key_frequencies
    NOTES = {
        '0': 0, 'e0': 20.60172, 'f0': 21.82676, 'f#0': 23.12465, 'g0': 24.49971, 'g#0': 25.95654, 'a0': 27.50000, 'a#0': 29.13524,
        'b0': 30.86771, 'c0': 32.70320, 'c#0': 34.64783, 'd0': 36.70810, 'd#0': 38.89087,
        'e1': 41.20344, 'f1': 43.65353, 'f#1': 46.24930, 'g1': 48.99943, 'g#1': 51.91309, 'a1': 55.00000, 'a#1': 58.27047,
        'b1': 61.73541, 'c1': 65.40639, 'c#1': 69.29566, 'd1': 73.41619, 'd#1': 77.78175,
        'e2': 82.40689, 'f2': 87.30706, 'f#2': 92.49861, 'g2': 97.99886, 'g#2': 103.8262, 'a2': 110.0000, 'a#2': 116.5409,
        'b2': 123.4708, 'c2': 130.8128, 'c#2': 138.5913, 'd2': 146.8324, 'd#2': 155.5635,
        'e3': 164.8138, 'f3': 174.6141, 'f#3': 184.9972, 'g3': 195.9977, 'g#3': 207.6523, 'a3': 220.0000, 'a#3': 233.0819,
        'b3': 246.9417, 'c3': 261.6256, 'c#3': 277.1826, 'd3': 293.6648, 'd#3': 311.1270,
        'e4': 329.6276, 'f4': 349.2282, 'f#4': 369.9944, 'g4': 391.9954, 'g#4': 415.3047, 'a4': 440.0000, 'a#4': 466.1638,
        'b4': 493.8833, 'c4': 523.2511, 'c#4': 554.3653, 'd4': 587.3295, 'd#4': 622.2540,
        'e5': 659.2551, 'f5': 698.4565, 'f#5': 739.9888, 'g5': 783.9909, 'g#5': 830.6094, 'a5': 880.0000, 'a#5': 932.3275,
        'b5': 987.7666, 'c5': 1046.502, 'c#5': 1108.731, 'd5': 1174.659, 'd#5': 1244.508,
        'e6': 1318.510, 'f6': 1396.913, 'f#6': 1479.978, 'g6': 1567.982, 'g#6': 1661.219, 'a6': 1760.000, 'a#6': 1864.655,
        'b6': 1975.533, 'c6': 2093.005, 'c#6': 2217.461, 'd6': 2349.318, 'd#6': 2489.016,
        'e7': 2637.020, 'f7': 2793.826, 'f#7': 2959.955, 'g7': 3135.963, 'g#7': 3322.438, 'a7': 3520.000, 'a#7': 3729.310,
        'b7': 3951.066, 'c7': 4186.009, 'c#7': 4434.922, 'd7': 4698.636, 'd#7': 4978.032,
    }

    common_timeline = np.linspace(0, DURATION_SECONDS, num=SOUND_ARRAY_LEN)

    def get_normed_sin(self, timeline, frequency):
        return self.MAX_AMPLITUDE * np.sin(2 * np.pi * frequency * timeline)

    def get_soundwave(self, timeline, note):
        return self.get_normed_sin(timeline, self.NOTES[note])

    def create_note(self, note="a4", name=None, timeline: np.dtype("float64") = None):
        if timeline is None:
            timeline = self.common_timeline

        sound_wave = self.get_soundwave(timeline, note).astype(np.int16)
        if name is None:
            file_name = f"{note}_sin.wav".replace("#", "s")
        else:
            file_name = f"{name}.wav"
        wavfile.write(file_name, self.SAMPLING_RATE, sound_wave)
        return sound_wave

    def read_wave_from_txt(self, file_path):
        return np.loadtxt(file_path)

    def print_wave_details(self, wave, sampling_rate=None):
        if not sampling_rate:
            sampling_rate = self.SAMPLING_RATE

        duration = len(wave) / sampling_rate
        min_amplitude = np.min(wave)
        max_amplitude = np.max(wave)
        mean_amplitude = np.mean(wave)
        std_amplitude = np.std(wave)
        wave_length = len(wave)

        print(f"Duration: {duration}")
        print(f"Min Amplitude: {min_amplitude}")
        print(f"Max Amplitude: {max_amplitude}")
        print(f"Mean Amplitude: {mean_amplitude}")
        print(f"Standard Deviation: {std_amplitude}")
        print(f"Wave Length: {wave_length}")

    def normalize_waves(self, *waves):
        min_length_wave = min([len(wave) for wave in waves])
        trimmed_waves = [wave[:min_length_wave] for wave in waves]
        min_amplitude = np.min(trimmed_waves)
        max_amplitude = np.max(trimmed_waves)
        return [(wave - min_amplitude) / (max_amplitude - min_amplitude) for wave in trimmed_waves]

    def save_wave(self, wave, file_name, file_type):
        if file_type == "WAV":
            wavfile.write(f"{file_name}.wav", self.SAMPLING_RATE, wave.astype(np.int16))
        else:
            np.savetxt(f"{file_name}.txt", wave)


if __name__ == "__main__":
    sw_factory = SoundWaveFactory()
    a4 = sw_factory.create_note()
    a1 = sw_factory.create_note(note='a1')

    sw_factory.save_wave(a1, 'a1_wave', 'txt')
    sw_factory.save_wave(a4, 'a4_wave', 'txt')

    a4_wave = sw_factory.read_wave_from_txt('a1_wave.txt')
    a1_wave = sw_factory.read_wave_from_txt('a4_wave.txt')

    sw_factory.print_wave_details(a4_wave)
    sw_factory.print_wave_details(a1_wave)

    print(sw_factory.normalize_waves(a1_wave, a4_wave))

    sw_factory.save_wave(a4_wave, 'a4_wave_as_wav', 'WAV')
