% modular synthesis
% 8 Jul 2024
pkg load signal

sampling_freq = 48000;
total_time    = 0:1/sampling_freq:10;

frequency_center = 110;
frequency_mod    = 110;

angular_freq_center = 2*pi*frequency_center;
angular_freq_mod    = 2*pi*frequency_mod;

a = 21 * linspace(1,0,length(total_time));

sound_wave = cos(angular_freq_center * total_time + a .* cos(angular_freq_mod * total_time));

figure(1)
specgram(sound_wave, 4096, sampling_freq);
