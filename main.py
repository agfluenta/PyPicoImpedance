# minimal_pico_voltage.py
from ctypes import byref, c_int16, c_int32, c_float
import numpy as np

from picosdk.ps3000a import ps3000a as ps
from picosdk.functions import adc2mV, assert_pico_ok

NUM_SAMPLES = 1000
CH = ps.PS3000A_CHANNEL["PS3000A_CHANNEL_A"]
RANGE = ps.PS3000A_RANGE["PS3000A_5V"]  # ±5 V, adjust if needed

handle = c_int16()
assert_pico_ok(ps.ps3000aOpenUnit(byref(handle), None))

try:
    # Enable ChA, DC, ±5 V
    assert_pico_ok(ps.ps3000aSetChannel(
        handle, CH, 1,  # enabled
        ps.PS3000A_COUPLING["PS3000A_DC"],
        RANGE, 0.0
    ))

    # No trigger (immediate)
    assert_pico_ok(ps.ps3000aSetSimpleTrigger(handle, 0, CH, 0, 0, 0, 0))

    # Pick a valid timebase
    tb = 8
    dt_ns = c_float()
    max_samps = int()
    while ps.ps3000aGetTimebase2(
            handle,
            tb,
            NUM_SAMPLES,
            byref(dt_ns),
            max_samps,
            0,  # segment index
            0  # reserved
    ) != 0:
        tb += 1

    # Buffer
    buf = (c_int16 * NUM_SAMPLES)()
    assert_pico_ok(ps.ps3000aSetDataBuffer(
        handle, CH, buf, NUM_SAMPLES, 0,
        ps.PS3000A_RATIO_MODE["PS3000A_RATIO_MODE_NONE"]
    ))

    # Acquire
    assert_pico_ok(ps.ps3000aRunBlock(handle, 0, NUM_SAMPLES, tb, None, 0, None, None))
    ready = c_int16(0)
    while not ready.value:
        assert_pico_ok(ps.ps3000aIsReady(handle, byref(ready)))

    n = c_int32(NUM_SAMPLES)
    over = c_int16()
    assert_pico_ok(ps.ps3000aGetValues(
        handle, 0, byref(n), 1,
        ps.PS3000A_RATIO_MODE["PS3000A_RATIO_MODE_NONE"],
        0, byref(over)
    ))

    # Convert to volts
    max_adc = c_int16()
    assert_pico_ok(ps.ps3000aMaximumValue(handle, byref(max_adc)))
    mv = adc2mV(buf, RANGE, max_adc)
    v = np.asarray(mv, dtype=np.float64) / 1000.0

    dt = dt_ns.value * 1e-9
    print(f"dt = {dt*1e6:.3f} µs  samples = {n.value}  overflow = {over.value}")
    print(f"Vavg = {v.mean():.6f} V  Vmin = {v.min():.6f} V  Vmax = {v.max():.6f} V")

finally:
    ps.ps3000aStop(handle)
    ps.ps3000aCloseUnit(handle)
