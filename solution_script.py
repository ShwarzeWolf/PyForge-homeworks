import os
import re

import numpy as np
from scipy.io import wavfile

SAMPLING_RATE = 44100
DURATION_SECONDS = 5
SOUND_ARRAY_LEN = SAMPLING_RATE * DURATION_SECONDS
MAX_AMPLITUDE = 2 ** 13

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

def set_custom_common_timeline(duration: float):
    sound_array_len = int(SAMPLING_RATE * duration)
    custom_timeline = np.linspace(0, duration, num=sound_array_len, endpoint=False)
    return custom_timeline

class SoundWave:
    """
    A factory class for creating, managing and manipulating sound waves.

    Attributes:
        note_str (str): The normalized sound wave data.
        sound_wave (numpy.ndarray): the sound wave array.
    Methods:
        get_details(): Returns the details of the sound wave.
    """

    def __init__(self, note_str, sound_wave):
        self.note_str = note_str
        self.sound_wave = sound_wave

    def get_details(self):
        return f"Note: {self.note_str}, Sound wave: {self.sound_wave}"


class SoundWaveFactory:
    """
    A factory class for creating, managing and manipulating sound waves.

    Attributes:
        normalized_wave (numpy.ndarray): The normalized sound wave data.
        sampling_rate (int): The number of samples per second (default is 44100).
        duration_seconds (float): The duration of the sound wave in seconds.
        max_amplitude (int): The maximum amplitude for the sound wave (default is 32767).

    Methods:
        create_sound_wave(): Creates a sound wave for the specified note.
        normalize_sound_waves(*waves): Normalizes multiple sound waves by length and amplitude.
        save_wave(type: str): Saves the sound wave as a WAV or TXT file.
        read_wave_from_txt(filename: str): Reads a sound wave from a TXT file.
        combine_waves(*sound_waves): Combines multiple sound waves into a single wave.
        convert_wave(sine_wave, wave_type): Converts a sine wave to a triangular or square wave.
        create_triangular_wave(frequency, duration): Creates a triangular wave with the specified frequency and duration.
        create_square_wave(frequency, duration): Creates a square wave with the specified frequency and duration.
        read_notes_string(melody_str): Reads a text string of notes and generates a melody.
        generate_group_wave(notes, duration): Generates a group wave for a list of notes and duration.

    """
    def __init__(self):
        self.sampling_rate = SAMPLING_RATE
        self.duration_seconds = DURATION_SECONDS
        self.sound_array_len = SAMPLING_RATE * DURATION_SECONDS
        self.max_amplitude = MAX_AMPLITUDE

    def create_sound_wave(self, note_str, duration=None):
        """
        Creates a sound wave for the specified musical note.

        This method generates a sound wave based on the given note string
        and optional duration. If no duration is provided, a default
        duration is used.

        Args:
            note_str (str): The musical note to generate (e.g., 'A4').
            duration (float, optional): The duration of the note in seconds. Defaults to None.

        Returns:
            SoundWave: An instance of the SoundWave class containing the generated sound wave.
        """


        if duration is not None:
            timeline = set_custom_common_timeline(duration)
        else:
            timeline = common_timeline
        frequency = NOTES[note_str]
        sound_wave = self.get_soundwave(timeline, note_str)
        return SoundWave(note_str, sound_wave)

    def get_soundwave(self, timeline, note):
        """
        Generates a sound wave based on the provided timeline and musical note.

        Args:
            timeline (numpy.ndarray): The time points at which the wave is sampled.
            note (str): The musical note for which the sound wave is generated.

        Returns:
            numpy.ndarray: The generated sound wave as a normalized sine wave.
        """
        return self.get_normed_sin(timeline, NOTES[note])

    def get_normed_sin(self, timeline, frequency):
        """
        Generates a normalized sine wave for the given frequency over the specified timeline.

        Args:
            timeline (numpy.ndarray): The time points at which the wave is sampled.
            frequency (float): The frequency of the sine wave to be generated.

        Returns:
            numpy.ndarray: The normalized sine wave.
        """
        return self.max_amplitude * np.sin(2 * np.pi * frequency * timeline)

    def normalize_sound_waves(self, *waves):
        """
        Normalizes multiple sound waves to the same length and amplitude.

        Args:
            *waves (numpy.ndarray): The sound waves to be normalized.

        Returns:
            list: A list of normalized sound waves.
        """
        min_length = min(len(wave) for wave in waves)
        normalized_waves = []

        for wave in waves:
            truncated_wave = wave[:min_length]
            max_amplitude = np.max(np.abs(truncated_wave))
            normalized_wave = (truncated_wave / max_amplitude) * self.max_amplitude
            normalized_waves.append(normalized_wave)

        return normalized_waves

    def perform_adsr_mods(self, wave: SoundWave, attack_time, decay_time, sustain_level, release_time):
        sample_rate = SAMPLING_RATE
        sound_wave = wave.sound_wave
        attack_samples = int(attack_time * sample_rate)
        decay_samples = int(decay_time * sample_rate)
        sustain_samples = int((len(sound_wave) / sample_rate) - (attack_time + decay_time + release_time))
        release_samples = int(release_time * sample_rate)

        envelope = np.zeros(len(sound_wave))

        for i in range(attack_samples):
            envelope[i] = (i / attack_samples)

        for i in range(decay_samples):
            envelope[attack_samples + i] = 1 - (i / decay_samples) * (1 - sustain_level)

        envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain_level

        for i in range(release_samples):
            envelope[attack_samples + decay_samples + sustain_samples + i] = sustain_level * (1 - (i / release_samples))

        modulated_sound = sound_wave * envelope
        wave.sound_wave = modulated_sound
        return wave

    def read_wave_from_txt(self, filename):
        """
        Reads a sound wave from a text file and creates a SoundWave instance.

        Args:
            filename (str): The name of the text file to read the sound wave from.

        Returns:
            SoundWave: An instance of the SoundWave class containing the sound wave.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file contains improperly formatted data.
        """
        note_str = filename[:-4]  # get rid of ".txt" in filename to be left with note name
        try:
            sound_wave = self.get_soundwave(common_timeline, note_str)
            return SoundWave(note_str, sound_wave)
        except FileNotFoundError:
            print(f"File {note_str} not found.")
        except ValueError:
            print(f"File {note_str} contains improperly formatted data.")

    def combine_waves(self, *soundwaves):
        """
        Combines multiple SoundWave instances into a single sound wave.

        Args:
            *soundwaves (SoundWave): The sound waves to be combined.

        Returns:
            SoundWave: A new SoundWave instance containing the combined sound wave.
        """
        combined_wave = np.concatenate([sw.sound_wave for sw in soundwaves])
        return SoundWave('combined', combined_wave)

    def convert_wave(self, sine_wave, wave_type):
        """
        Converts a sine wave to a specified wave type (triangular or square).

        Args:
            sine_wave (numpy.ndarray): The sine wave to be converted.
            wave_type (str): The type of wave to convert to ('triangular' or 'square').

        Returns:
            numpy.ndarray: The converted wave.

        Raises:
            ValueError: If an unsupported wave type is specified.
        """
        if wave_type == 'triangular':
            return self.create_triangular_wave(sine_wave)
        elif wave_type == 'square':
            return self.create_square_wave(sine_wave)
        else:
            raise ValueError("Unsupported wave type. Use 'triangular' or 'square'.")

    def create_triangular_wave(self, sine_wave):
        """
        Creates a triangular wave for a specified frequency and duration.

        Args:
            frequency (float): The frequency of the triangular wave.
            duration (float): The duration of the triangular wave in seconds.

        Returns:
            numpy.ndarray: The generated triangular wave.
        """
        normalized_sine = sine_wave / np.max(np.abs(sine_wave))

        triangular_wave = 2 * np.abs(2 * (normalized_sine - np.floor(normalized_sine + 0.5))) - 1

        return triangular_wave

    def create_square_wave(self, sine_wave):
        """
        Creates a square wave for a specified frequency and duration.

        Args:
            frequency (float): The frequency of the square wave.
            duration (float): The duration of the square wave in seconds.

        Returns:
            numpy.ndarray: The generated square wave.
        """
        normalized_sine = sine_wave / np.max(np.abs(sine_wave))

        # Convert the normalized sine wave to square wave
        square_wave = 0.5 * (1 + np.sign(normalized_sine))
        return square_wave

    def save_wave(self, type: str, wave_instance: SoundWave):
        """
        Saves a sound wave to a file in the specified format (WAV or NumPy array).

        Args:
            type (str): The file format to save the wave ('WAV' or other).
            wave_instance (SoundWave): The SoundWave instance containing the wave to save.

        Returns:
            None
        """
        wav_folder = "wav_files"
        npy_folder = "npy_files"

        if not os.path.exists(wav_folder):
            os.makedirs(wav_folder)

        if not os.path.exists(npy_folder):
            os.makedirs(npy_folder)

        if type == 'WAV':
            wav_file_path = os.path.join(wav_folder, f"{wave_instance.note_str}.wav")
            wavfile.write(wav_file_path, SAMPLING_RATE, wave_instance.sound_wave)
            print(f"Saved wave as {wav_file_path}")

        txt_file_path = os.path.join(npy_folder, f"{wave_instance.note_str}.txt")
        np.savetxt(txt_file_path, wave_instance.sound_wave)
        print(f"Saved wave as {txt_file_path}")

    def read_notes_string(self, melody_str):
        """
        Parses a string of notes and generates a melody by creating corresponding sound waves.

        Args:
            melody_str (str): A string representing the melody, with notes and durations.

        Returns:
            numpy.ndarray: A concatenated array of sound waves representing the melody.
        """
        tokens = re.findall(r'\(.*?\)|\S+', melody_str)

        melody_sequence = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.startswith('(') and token.endswith(')'):
                notes_in_group = token[1:-1].split()

                i += 1
                duration = float(tokens[i][:-1])

                group_wave = self.generate_group_wave(notes_in_group, duration)
                melody_sequence.append(group_wave)
            else:
                note_str = token
                i += 1
                duration = float(tokens[i][:-1])

                sound_wave = self.create_sound_wave(note_str, duration)
                melody_sequence.append(sound_wave.sound_wave)

            i += 1

        return np.concatenate(melody_sequence)

    def generate_group_wave(self, notes, duration):
        """
        Generates a combined sound wave for a group of notes over a specified duration.

        Args:
            notes (list): A list of note strings to generate waves for.
            duration (float): The duration for which the group wave is generated.

        Returns:
            numpy.ndarray: The combined sound wave for the group of notes.
        """
        group_wave = np.zeros(int(SAMPLING_RATE * duration))
        for note in notes:
            note_wave = self.create_sound_wave(note, duration).sound_wave
            group_wave += note_wave
        return group_wave

    def save_melody(self, melody, filename):
        wav_folder = "wav_files"

        # Check if the folder exists, if not, create it
        if not os.path.exists(wav_folder):
            os.makedirs(wav_folder)

        # Full path for the WAV file
        wav_file_path = os.path.join(wav_folder, f"{filename}.wav")

        # Save the melody to the WAV file
        wavfile.write(wav_file_path, SAMPLING_RATE, melody)
        print(f"Saved wave as {wav_file_path}")