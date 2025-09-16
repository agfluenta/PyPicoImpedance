# Impedance analyzer (PS3000A) — swept sine, 30–60 kHz, 2 Vpp, ~2000 pts/dec
# Channels: CH A = Vdut, CH B = Vshunt. Current I = Vshunt / Rs, Z = Vdut / I
# Requires: picosdk Python wrappers installed and PS3000A driver.
# Tested pattern follows Pico's block-mode examples (OpenUnit, SetChannel, GetTimebase2, RunBlock, SetDataBuffers, GetValues, MaximumValue).

import ctypes
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # or any backend you use
import matplotlib.pyplot as plt

from picosdk.ps3000a import ps3000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

# ---------------------------
# User parameters
# ---------------------------
F_START = 30_000.0
F_STOP  = 60_000.0
PTS_PER_DECADE = 2000
N_POINTS = int(round(PTS_PER_DECADE * math.log10(F_STOP/F_START)))  # ~602
VPP_SIGNAL_UV = 2_000_000   # 2 Vpp in microvolts for SigGen
RS_OHMS = 100.0             # precision shunt; adjust to your DUTs
CH_RANGE_A = 8              # PS3000A_10V (example); adjust to your expected Vdut
CH_RANGE_B = 7              # PS3000A_5V (example); adjust to your expected Vshunt
COUPLING_DC = 1             # PS3000A_DC
N_CYCLES_CAPTURE = 16       # total captured cycles per tone (before trimming)
SETTLE_CYCLES = 2           # cycles to discard at start of record
MIN_SAMP_PER_CYCLE = 64     # ≥64 samples/cycle for clean phase
TIMEBASE_START_GUESS = 2    # will iterate upward until constraints are met

# ---------------------------
# Helpers
# ---------------------------
def logspace_frequencies(f_start, f_stop, n):
    return np.geomspace(f_start, f_stop, n)

def choose_timebase_for_f(handle, f_hz, n_cycles, min_samples_per_cycle, tb_guess=TIMEBASE_START_GUESS):
    """
    Iterate timebase until both:
      - samples per cycle >= min_samples_per_cycle
      - total samples >= n_cycles * samples_per_cycle
    Returns (timebase, n_total_samples, dt_ns)
    """
    # We will try a set of candidate total samples per tone (powers of two-ish), then find a timebase that satisfies min samp/cycle
    # Start from a practical target duration: Tcap = n_cycles / f
    T_target = n_cycles / f_hz
    # We'll aim for total samples near fs*T_target with fs derived from dt_ns once we know it.
    # Practical approach: iterate timebase and pick a total_samples that gives >= MIN_SAMP_PER_CYCLE at f_hz and >= n_cycles*MIN_SAMP_PER_CYCLE total
    # Use a modest range of total_samples to try; increase if needed.
    for total_samples in [4096, 8192, 16384, 32768, 65536]:
        tb = tb_guess
        while tb < 25_000:  # arbitrary safety ceiling
            timeIntervalns = ctypes.c_float()
            returnedMaxSamples = ctypes.c_int32()
            status = ps.ps3000aGetTimebase2(handle, tb, total_samples,
                                            ctypes.byref(timeIntervalns),
                                            1, ctypes.byref(returnedMaxSamples), 0)
            if status == 0:  # PICO_OK
                dt_ns = float(timeIntervalns.value)
                fs = 1e9 / dt_ns
                spc = fs / f_hz  # samples per cycle
                if spc >= min_samples_per_cycle:
                    # ensure our requested total_samples is not above driver limit
                    n_tot = min(total_samples, int(returnedMaxSamples.value))
                    if n_tot >= int(n_cycles * min_samples_per_cycle):
                        return tb, n_tot, dt_ns
            tb += 1
    raise RuntimeError("Could not find a suitable timebase / record length for f = %.1f Hz" % f_hz)

def setup_scope(handle):
    # Open & power source handling already done by caller; set up channels
    # CH A: Vdut
    st = ps.ps3000aSetChannel(handle, 0, 1, COUPLING_DC, CH_RANGE_A, 0.0)
    assert_pico_ok(st)
    # CH B: Vshunt
    st = ps.ps3000aSetChannel(handle, 1, 1, COUPLING_DC, CH_RANGE_B, 0.0)
    assert_pico_ok(st)
    # Disable triggering (free-run). We'll capture steady-state sine anyway.
    st = ps.ps3000aSetSimpleTrigger(handle, 0, 0, 0, 0, 0, 0)
    assert_pico_ok(st)

def set_siggen_sine(handle, freq_hz, vpp_uV=2_000_000):
    """
    Emit a steady sine at freq_hz with 0 offset using ps3000aSetSigGenBuiltIn.
    All scalar args must be native Python ints/floats (not ctypes instances).
    """
    offset_uV = 0
    waveType = 0        # PS3000A_SINE
    startFrequency = float(freq_hz)
    stopFrequency  = float(freq_hz)
    increment      = 0.0
    dwellTime      = 0.0
    sweepType = 0      # PS3000A_UP (ignored for single-tone)
    operation = 0      # PS3000A_ES_OFF
    shots = 0
    sweeps = 0
    triggerType = 0    # PS3000A_SIGGEN_TRIG_NONE
    triggerSource = 0  # PS3000A_SIGGEN_NONE
    extInThreshold = 0 # int16

    st = ps.ps3000aSetSigGenBuiltIn(
        handle,
        int(offset_uV),
        int(vpp_uV),
        int(waveType),
        startFrequency, stopFrequency,
        increment, dwellTime,
        int(sweepType),
        int(operation),
        int(shots),
        int(sweeps),
        int(triggerType),
        int(triggerSource),
        int(extInThreshold)
    )
    assert_pico_ok(st)



def stop_siggen(handle):
    # Zero the amplitude to stop output
    set_siggen_sine(handle, 1.0, vpp_uV=0)

def acquire_block_two_channels(handle, timebase, n_samples):
    # Prepare buffers
    bufA_max = (ctypes.c_int16 * n_samples)()
    bufA_min = (ctypes.c_int16 * n_samples)()
    bufB_max = (ctypes.c_int16 * n_samples)()
    bufB_min = (ctypes.c_int16 * n_samples)()

    # Set buffers (A=0, B=1)
    st = ps.ps3000aSetDataBuffers(handle, 0, ctypes.byref(bufA_max), ctypes.byref(bufA_min), n_samples, 0, 0)
    assert_pico_ok(st)
    st = ps.ps3000aSetDataBuffers(handle, 1, ctypes.byref(bufB_max), ctypes.byref(bufB_min), n_samples, 0, 0)
    assert_pico_ok(st)

    # Run block
    timeIndisposedMs = ctypes.c_int32()
    st = ps.ps3000aRunBlock(handle, 0, n_samples, timebase, 1, ctypes.byref(timeIndisposedMs), 0, None, None)
    assert_pico_ok(st)

    # Wait ready
    ready = ctypes.c_int16(0)
    while ready.value == 0:
        ps.ps3000aIsReady(handle, ctypes.byref(ready))

    # Get data
    cSamples = ctypes.c_int32(n_samples)
    overflow = ctypes.c_int16()
    st = ps.ps3000aGetValues(handle, 0, ctypes.byref(cSamples), 0, 0, 0, ctypes.byref(overflow))
    assert_pico_ok(st)

    # Stop acquisition engine (not strictly necessary between blocks, but clean)
    st = ps.ps3000aStop(handle)
    assert_pico_ok(st)

    return np.frombuffer(bufA_max, dtype=np.int16, count=n_samples), np.frombuffer(bufB_max, dtype=np.int16, count=n_samples)

def counts_to_volts(handle, counts_array, ch_range):
    # counts_array can be a numpy array or ctypes buffer
    maxADC = ctypes.c_int16()
    st = ps.ps3000aMaximumValue(handle, ctypes.byref(maxADC))
    assert_pico_ok(st)

    # Ensure adc2mV sees *plain Python ints*, not numpy.int16, to avoid overflow
    mv_list = adc2mV(counts_array.tolist(), ch_range, maxADC)   # -> list of mV
    mv = np.asarray(mv_list, dtype=np.float64)                  # -> float array
    return mv * 1e-3                                            # mV -> V


def trim_settle_cycles(x, fs, f_tone, settle_cycles):
    n_settle = int(round(settle_cycles * fs / f_tone))
    return x[n_settle:]

def single_bin_phasor(x, fs, f_tone):
    """
    Compute complex phasor at f_tone by correlation with e^{-j2πft} over the record.
    Returns complex amplitude (linear volts).
    """
    n = len(x)
    t = np.arange(n) / fs
    ref = np.exp(-1j * 2 * np.pi * f_tone * t)
    # Least-squares demodulation (complex)
    ph = (2.0 / n) * np.dot(ref, x)  # scale ~ RMS-consistent for pure sine
    return ph

def bode_plot(freqs, Z_complex):
    mag = np.abs(Z_complex)
    pha = np.angle(Z_complex, deg=True)

    fig1 = plt.figure()
    plt.semilogx(freqs, mag)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|Z| (Ohms)")
    plt.title("Impedance magnitude")

    fig2 = plt.figure()
    plt.semilogx(freqs, pha)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title("Impedance phase")

    plt.show()

def main():
    status = {}
    chandle = ctypes.c_int16()

    # Open device
    status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)
    try:
        assert_pico_ok(status["openunit"])
    except:  # handle power source negotiation
        powerstate = status["openunit"]
        if powerstate in (282, 286):
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, powerstate)
            assert_pico_ok(status["ChangePowerSource"])
        else:
            raise

    h = chandle
    setup_scope(h)

    # Frequency plan
    freqs = logspace_frequencies(F_START, F_STOP, N_POINTS)
    Z = np.zeros_like(freqs, dtype=np.complex128)

    # We'll reuse one timebase if it suffices worst-case (60 kHz). If not, we adapt per tone.
    # First choose for F_STOP
    tb, n_samp, dt_ns = choose_timebase_for_f(h, F_STOP, N_CYCLES_CAPTURE, MIN_SAMP_PER_CYCLE, TIMEBASE_START_GUESS)
    fs = 1e9 / dt_ns

    # Start sweep
    for i, f in enumerate(freqs):
        # Check if current timebase still meets constraints; if not, pick a new one
        spc = fs / f
        n_tot_need = int(N_CYCLES_CAPTURE * MIN_SAMP_PER_CYCLE)
        if (spc < MIN_SAMP_PER_CYCLE) or (n_samp < n_tot_need):
            tb, n_samp, dt_ns = choose_timebase_for_f(h, f, N_CYCLES_CAPTURE, MIN_SAMP_PER_CYCLE, tb)
            fs = 1e9 / dt_ns

        # Program SigGen for single steady tone at f, 2 Vpp
        set_siggen_sine(h, f, VPP_SIGNAL_UV)
        # Small wait to ensure generator has settled (fraction of a cycle is enough)
        time.sleep(max(3.0/f, 1e-4))

        # Acquire block on both channels
        a_counts, b_counts = acquire_block_two_channels(h, tb, n_samp)

        # Convert to volts
        vA = counts_to_volts(h, a_counts, CH_RANGE_A)  # Vdut
        vB = counts_to_volts(h, b_counts, CH_RANGE_B)  # Vshunt

        # Trim initial settle cycles
        vA = trim_settle_cycles(vA, fs, f, SETTLE_CYCLES)
        vB = trim_settle_cycles(vB, fs, f, SETTLE_CYCLES)

        # Recompute fs-aligned t (length changed)
        # (We pass fs and f directly to the demod; length is implicit via arrays)
        VA = single_bin_phasor(vA, fs, f)
        VB = single_bin_phasor(vB, fs, f)

        I = VB / RS_OHMS
        Z[i] = VA / I

    # Stop generator and close
    stop_siggen(h)
    status["stop"] = ps.ps3000aStop(h)
    assert_pico_ok(status["stop"])
    status["close"] = ps.ps3000aCloseUnit(h)
    assert_pico_ok(status["close"])

    # Plot
    bode_plot(freqs, Z)

    # Also return arrays if run as a module
    return freqs, Z

if __name__ == "__main__":
    main()
