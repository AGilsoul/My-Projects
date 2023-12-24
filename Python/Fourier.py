import math

import numpy as np
import random
from typing import Union
import matplotlib.pyplot as plt


# ComplexNumber class, represents a complex number a + bi, where a is the real part, and b is the imaginary part
# Also computes phase and modulus of the number
class ComplexNumber:
    def __init__(self, a: float, b: float):
        self.Re = a
        self.Im = b
        self.modulus = 0
        self.phase = 0
        self.compute_phase()
        self.compute_modulus()

    # computes phase of the complex number
    def compute_phase(self) -> None:
        self.phase = np.arctan2(self.Im, self.Re)

    # computes modulus (magnitude) of the complex number
    def compute_modulus(self) -> None:
        self.modulus = np.sqrt(self.Re**2 + self.Im**2)

    # overload + operator between two complex numbers
    def __add__(self, other):
        return ComplexNumber(self.Re + other.Re, self.Im + other.Im)

    # overload += operator between two complex numbers
    def __iadd__(self, other):
        self.Re += other.Re
        self.Im += other.Im
        self.compute_phase()
        self.compute_modulus()
        return self

    # overload * operator between complex number and a real number
    def __mul__(self, other: float):
        return ComplexNumber(self.Re * other, self.Im * other)

    # overload / operator, divides each complex component by a real number
    def __truediv__(self, other: float):
        return ComplexNumber(self.Re / other, self.Im / other)

    def __str__(self) -> str:
        return f'{self.Re} + {self.Im}i'


# Discrete Signal class, stores discrete points and the time steps at which they occur
class DiscreteSignal:
    def __init__(self, time: Union[list, np.array], samples: Union[list, np.array], dt: float):
        self.time = time
        self.samples = samples
        self.dt = dt
        if dt != time[1] - time[0]:
            raise Exception('dt not consistent')

    # plots the discrete signal on a given axis
    def plot(self, axis, label_p='', c_p='') -> None:
        if label_p == '' or c_p == '':
            axis.plot(self.time, self.samples)
        else:
            axis.plot(self.time, self.samples, label=label_p, c=c_p)


# Signal class, represents a cosine wave with some amplitude, frequency, and phase
class Signal:
    def __init__(self, amplitude: float, frequency: float, phase: float):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase / (2 * np.pi)

    # evaluates signal at some time t
    def eval(self, t: float) -> float:
        return self.amplitude * np.cos(2 * np.pi * (self.frequency * t + self.phase))

    # plots the signal on a given axis
    def plot(self, axis, t_eval: Union[list, np.array], label_p='', c_p='') -> None:
        vals = [self.eval(t) for t in t_eval]
        if label_p == '' or c_p == '':
            axis.plot(t_eval, vals)
        else:
            axis.plot(t_eval, vals, label=label_p, c=c_p)

    def __str__(self) -> str:
        return f'amp: {self.amplitude}, freq: {self.frequency}, phase: {self.phase}'


# Approximation for dirac delta
def dirac_delta(x: float, a=1e-2) -> float:
    return (1 / (np.abs(a) * np.sqrt(np.pi))) * np.exp(-(x/a)**2)


# discrete fourier transform for a range of frequencies
def transform_signal(signal: DiscreteSignal, max_freq: int) -> list:
    sampling_freq = 1 / signal.dt  # sampling frequency of discrete signal

    # if max frequency to compute is above Nyquist limit, set frequency range to go up to Nyquist limit
    if 2 * max_freq >= sampling_freq:
        frequencies = range(0, int(sampling_freq/2)+1)
    else:
        frequencies = range(0, int(max_freq) + 1)
    # list of 0 valued complex numbers for each frequency
    transformed = [ComplexNumber(0, 0) for _ in frequencies]
    # for every frequency in the spectrum
    for frequency in range(len(frequencies)):
        # calculate power that e is raised to, multiplied by i
        b_n = -2 * np.pi * frequencies[frequency] / sampling_freq
        for sample in range(len(signal.samples)):
            cur_b_n = b_n * sample
            # Euler identity, split e^ib_n=cos(b_n) + isin(b_n)
            Re = signal.samples[sample] * np.cos(cur_b_n)
            Im = signal.samples[sample] * np.sin(cur_b_n)
            cur_wave = ComplexNumber(Re, Im)  # current component wave as complex number
            transformed[frequency] += cur_wave  # add to complex number for current frequency

    # Multiply complex components for each frequency by 2 to account for duplication above Nyquist limit
    # Need to average over all samples, so divide by number of samples
    transformed = [t * 2 / len(signal.samples) for t in transformed]
    # Construct Signal objects for each frequency
    print(max(transformed, key=lambda t: t.modulus))
    signals = [Signal(transformed[t].modulus, frequencies[t], transformed[t].phase) for t in range(len(transformed))]

    return signals


# generates a random signal composed of smaller signals given a range of amplitudes and frequencies
def rand_signal(num_signals: int, amplitude_range: Union[tuple, list, np.array], frequency_range: Union[tuple, list, np.array], time_scale: Union[int, float], num_pts: int, shift=True) -> DiscreteSignal:
    t = np.linspace(0, time_scale, num_pts)
    base_signal = [0 for _ in range(num_pts)]
    new_signal = base_signal
    # generate each random signal, then sum to generate final random signal
    for i in range(num_signals):
        cur_amp = random.uniform(amplitude_range[0], amplitude_range[1])
        cur_freq = random.randint(frequency_range[0], frequency_range[1])
        if shift:
            cur_phase = random.uniform(0, np.pi / 2)
            # cur_phase = 3/4
        else:
            cur_phase = 0
        print(f'Original signal: amp: {cur_amp}, freq: {cur_freq}, phase: {cur_phase}')
        cur_signal = cur_amp * np.cos(2 * np.pi * (t * cur_freq + cur_phase))
        new_signal += cur_signal

    new_signal = DiscreteSignal(t, new_signal, t[1] - t[0])
    return new_signal


# plots results of the discrete fourier transform
def plot_dft(original_signal: DiscreteSignal, component_signals: Union[list, np.array], t_scale: Union[list, tuple]) -> None:
    fig = plt.figure()
    wave_axis = fig.add_subplot(2, 2, 1)  # axis for displaying sum of important waves from fourier transform and original signal
    freq_axis = fig.add_subplot(2, 2, 2)  # axis for displaying frequency spectrum
    comp_axis = fig.add_subplot(2, 2, 3)  # axis for displaying all component waves from fourier transform
    t_eval = np.linspace(0, t_scale[-1], 1000)  # time steps to evaluate at

    # plot each component signal from dft
    for signal in component_signals:
        signal.plot(comp_axis, t_eval)

    # sum each component signal
    summed_components = [sum([signal.eval(t) for signal in component_signals]) for t in t_eval]

    # scatter plot of frequency spectrum
    freq_axis.scatter([signal.frequency for signal in component_signals],
                      [signal.amplitude for signal in component_signals])
    # plot summed component signals and the original signal
    wave_axis.plot(t_eval, summed_components, label='summed components', c='green')
    original_signal.plot(wave_axis, label_p='original', c_p='blue')
    # show legend
    wave_axis.legend()
    plt.show()


# performs a discrete fourier transform on signal composed of cosine waves with no phase shift
def symmetric_dft() -> None:
    t_scale = [0, 1]  # time scale for sampling
    signal = rand_signal(20, [0, 5], [1, 10], t_scale[-1], 10000, shift=False)  # generates random signal composed of 20 cosine waves
    component_signals = transform_signal(signal, 10)  # performs discrete fourier transform on random signal

    max_comp = max(component_signals, key=lambda p: p.amplitude)  # for printing out frequency of wave which contributes the most
    print(f'max signal: {max_comp}')
    plot_dft(signal, component_signals, t_scale)  # plots results


# same as above, but cosine waves can have phase shifts
def asymmetric_dft() -> None:
    t_scale = [0, 1]
    signal = rand_signal(10, [1, 5], [1, 10], t_scale[-1], 10000)
    component_signals = transform_signal(signal, 10)
    max_comp = max(component_signals, key=lambda p: p.amplitude)
    print(max_comp)
    print()
    plot_dft(signal, component_signals, t_scale)


def main() -> int:
    # symmetric_dft()  # perform discrete fourier transform on a sum of random cosine waves
    asymmetric_dft()  # perform discrete fourier transform on a sum of random shifted cosine waves
    return 0


if __name__ == '__main__':
    main()
