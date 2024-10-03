
from solution_script import *

note = SoundWaveFactory().create_sound_wave('a#2')
note2 = SoundWaveFactory().create_sound_wave('a#3')
note3 = SoundWaveFactory().create_sound_wave('a#4')


# TASK: a method to read a wave from .txt file
note_read_from_file = SoundWaveFactory().read_wave_from_txt('a#2.txt')
print(note_read_from_file.get_details())

# TASK: print the details of the note
print(note3.get_details())

# TASK: a method to normalize_sound_waves several waves: in both length (to the shortest file)
# and amplitude (according to the amplitude attribute)
norm_instances = SoundWaveFactory().normalize_sound_waves(note.sound_wave, note2.sound_wave, note3.sound_wave)

# TASK: a method to save wave into np.array txt by default and
# into WAV file if parameter "type='WAV'" is provided
SoundWaveFactory().save_wave('WAV', note)

# EXTRA TASK: extra: a method to switch sin waves into triangular or square waves;
note_for_conversion = SoundWaveFactory().create_sound_wave('a#2')
soundwave = SoundWaveFactory().get_soundwave(common_timeline, note_for_conversion.note_str)
note_triangular = SoundWaveFactory().convert_wave(soundwave, 'triangular')
note_for_conversion.sound_wave = note_triangular
print(note_for_conversion.get_details())


# EXTRA TASK: extra: a method to apply an ADSR envelope to a wave
print("CHECKING ADSR METHOD -------------------")
modified_note3_adsr = SoundWaveFactory().perform_adsr_mods(note3, 0.1, 0.1, 0.5, 0.1)
print(modified_note3_adsr.get_details())


# EXTRA TASK: extra: a method to combine the waves into a sequence of notes;
combined_wave = SoundWaveFactory().combine_waves(note, note2, note3)
SoundWaveFactory().save_wave('none', combined_wave)

# EXTRA TASK: extra: a method to read a text string of notes so that you class could generate a given melody;
# notes text could look like this: "g4 0.2s b4 0.2s (g3 d5 g5) 0.5s"
# the melody of Am I Dreaming - Roisee, Metro Boomin, A$AP Rocky
melody_str = "a4 0.4s e4 0.2s f4 0.4s d#4 0.4s c4 0.4s e4 0.2s g4 0.4s a4 0.8s"

melody = SoundWaveFactory().read_notes_string(melody_str)
SoundWaveFactory().save_melody(melody, "Am_I_Dreaming")

