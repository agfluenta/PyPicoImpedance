"""Impedance analyser built on top of the PicoScope 3000A driver.

This module drives the built-in signal generator of a PicoScope 3000A to
stimulate a device under test (DUT) through a known reference resistor while
capturing the response on two oscilloscope channels. The complex impedance of
the DUT is derived from the captured data by comparing the voltage across the
reference resistor with the voltage across the DUT.

The default configuration performs a logarithmic sweep between 30 kHz and
60 kHz using 2000 points per decade, which yields 750 frequency points across
the requested range. A 2 Vpp sine excitation is produced for each frequency
before the scope acquires a block of samples for analysis.

The script can be executed directly and provides options for storing the sweep
to CSV as well as plotting a bode plot of the measured impedance.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

try:
    matplotlib.use("Qt5Agg")
except Exception:  # pragma: no cover - backend selection depends on system
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from picosdk.errors import PicoSDKCtypesError
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.ps3000a import ps3000a as ps


@dataclass
class ImpedancePoint:
    """Container for a single impedance measurement."""

    frequency_hz: float
    reference_voltage: complex
    dut_voltage: complex
    impedance: complex
    reference_resistance: float

    @property
    def impedance_magnitude(self) -> float:
        """Return the impedance magnitude in ohms."""

        return abs(self.impedance)

    @property
    def impedance_phase_deg(self) -> float:
        """Return the impedance phase in degrees."""

        return math.degrees(math.atan2(self.impedance.imag, self.impedance.real))

    @property
    def current_amps(self) -> float:
        """Return the stimulus current magnitude in amperes."""

        return abs(self.reference_voltage) / self.reference_resistance

    @property
    def voltage_ref_magnitude(self) -> float:
        return abs(self.reference_voltage)

    @property
    def voltage_dut_magnitude(self) -> float:
        return abs(self.dut_voltage)

    @property
    def voltage_ref_phase_deg(self) -> float:
        return math.degrees(math.atan2(self.reference_voltage.imag, self.reference_voltage.real))

    @property
    def voltage_dut_phase_deg(self) -> float:
        return math.degrees(math.atan2(self.dut_voltage.imag, self.dut_voltage.real))

    @property
    def phase_difference_deg(self) -> float:
        """Return DUT phase minus reference phase in degrees."""

        return self.voltage_dut_phase_deg - self.voltage_ref_phase_deg

    def to_csv_row(self) -> Dict[str, float]:
        """Return a dictionary that can be written directly to CSV."""

        return {
            "frequency_hz": self.frequency_hz,
            "impedance_real_ohms": self.impedance.real,
            "impedance_imag_ohms": self.impedance.imag,
            "impedance_magnitude_ohms": self.impedance_magnitude,
            "impedance_phase_deg": self.impedance_phase_deg,
            "current_amps": self.current_amps,
            "voltage_ref_magnitude_volts": self.voltage_ref_magnitude,
            "voltage_ref_phase_deg": self.voltage_ref_phase_deg,
            "voltage_dut_magnitude_volts": self.voltage_dut_magnitude,
            "voltage_dut_phase_deg": self.voltage_dut_phase_deg,
        }


class PicoImpedanceAnalyzer:
    """High-level controller that sweeps and measures impedance."""

    def __init__(
        self,
        freq_start_hz: float = 30_000.0,
        freq_stop_hz: float = 60_000.0,
        points_per_decade: int = 2000,
        voltage_pk_pk: float = 2.0,
        reference_resistance_ohms: float = 1_000.0,
        samples_per_waveform: int = 4096,
        cycles_per_waveform: float = 10.0,
        settle_time_s: float = 0.05,
        channel_range: int = 8,
        timebase_search_limit: int = 500,
    ) -> None:
        if freq_start_hz <= 0 or freq_stop_hz <= 0:
            raise ValueError("Frequencies must be positive values in hertz.")
        if freq_stop_hz <= freq_start_hz:
            raise ValueError("Stop frequency must be greater than start frequency.")

        self.freq_start_hz = float(freq_start_hz)
        self.freq_stop_hz = float(freq_stop_hz)
        self.points_per_decade = int(points_per_decade)
        self.voltage_pk_pk = float(voltage_pk_pk)
        self.reference_resistance = float(reference_resistance_ohms)
        self.samples_per_waveform = int(samples_per_waveform)
        self.cycles_per_waveform = float(cycles_per_waveform)
        self.settle_time_s = float(settle_time_s)
        self.channel_range = int(channel_range)
        self.timebase_search_limit = int(timebase_search_limit)

        if self.samples_per_waveform <= 0:
            raise ValueError("samples_per_waveform must be positive")
        if self.cycles_per_waveform <= 0:
            raise ValueError("cycles_per_waveform must be positive")

        self.status: Dict[str, int] = {}
        self.chandle = ctypes.c_int16()
        self.max_adc = ctypes.c_int16()
        self._frequencies = self._generate_frequencies()

    @property
    def frequencies(self) -> np.ndarray:
        return self._frequencies

    def _generate_frequencies(self) -> np.ndarray:
        decades = math.log10(self.freq_stop_hz) - math.log10(self.freq_start_hz)
        base_points = max(2, int(math.ceil(self.points_per_decade * decades)))
        # Ensure we honour the 2000 points/decade requirement with ~750 samples in range
        num_points = max(750, base_points)
        frequencies = np.logspace(
            math.log10(self.freq_start_hz),
            math.log10(self.freq_stop_hz),
            num=num_points,
        )
        logging.debug(
            "Prepared %d logarithmically spaced frequency points between %.2f kHz and %.2f kHz.",
            num_points,
            self.freq_start_hz / 1000.0,
            self.freq_stop_hz / 1000.0,
        )
        return frequencies

    # ------------------------------------------------------------------
    # PicoScope control helpers
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Open and configure the PicoScope instrument."""

        logging.info("Opening PicoScope 3000A device")
        self.status["open_unit"] = ps.ps3000aOpenUnit(ctypes.byref(self.chandle), None)
        try:
            assert_pico_ok(self.status["open_unit"])
        except PicoSDKCtypesError:
            power_state = self.status["open_unit"]
            logging.warning("Power change required by device (state %s)", power_state)
            if power_state in (282, 286):
                self.status["change_power"] = ps.ps3000aChangePowerSource(self.chandle, power_state)
                assert_pico_ok(self.status["change_power"])
            else:
                raise

        self._configure_channels()
        self.status["disable_trigger"] = ps.ps3000aSetSimpleTrigger(self.chandle, 0, 0, 0, 0, 0, 0)
        assert_pico_ok(self.status["disable_trigger"])
        self.status["maximum_value"] = ps.ps3000aMaximumValue(self.chandle, ctypes.byref(self.max_adc))
        assert_pico_ok(self.status["maximum_value"])

    def _configure_channels(self) -> None:
        """Enable channels A and B for differential measurement."""

        for channel in (0, 1):
            key = f"set_channel_{channel}"
            self.status[key] = ps.ps3000aSetChannel(self.chandle, channel, 1, 1, self.channel_range, 0.0)
            assert_pico_ok(self.status[key])

    def _configure_signal_generator(self, frequency_hz: float, pk_to_pk_voltage: Optional[float] = None) -> None:
        """Configure the built-in signal generator for a sine output."""

        amplitude = self.voltage_pk_pk if pk_to_pk_voltage is None else float(pk_to_pk_voltage)
        pk_to_pk_microvolts = int(round(amplitude * 1_000_000))
        self.status["siggen"] = ps.ps3000aSetSigGenBuiltIn(
            self.chandle,
            0,
            pk_to_pk_microvolts,
            0,  # PS3000A_SINE
            float(frequency_hz),
            float(frequency_hz),
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        assert_pico_ok(self.status["siggen"])

    def _find_timebase(self, total_samples: int, target_interval_ns: float) -> tuple[int, float]:
        """Find a timebase that provides at least the target sample interval."""

        last_good: Optional[tuple[int, float]] = None
        last_status: Optional[int] = None

        for timebase in range(1, self.timebase_search_limit + 1):
            time_interval_ns = ctypes.c_float()
            returned_max_samples = ctypes.c_int32()
            status_code = ps.ps3000aGetTimebase2(
                self.chandle,
                timebase,
                total_samples,
                ctypes.byref(time_interval_ns),
                1,
                ctypes.byref(returned_max_samples),
                0,
            )
            last_status = status_code
            if status_code == 0:
                last_good = (timebase, float(time_interval_ns.value))
                if time_interval_ns.value >= target_interval_ns:
                    break

        if last_good is None:
            assert_pico_ok(last_status if last_status is not None else -1)

        chosen_timebase, _ = last_good
        time_interval_ns = ctypes.c_float()
        returned_max_samples = ctypes.c_int32()
        self.status["get_timebase"] = ps.ps3000aGetTimebase2(
            self.chandle,
            chosen_timebase,
            total_samples,
            ctypes.byref(time_interval_ns),
            1,
            ctypes.byref(returned_max_samples),
            0,
        )
        assert_pico_ok(self.status["get_timebase"])
        return chosen_timebase, float(time_interval_ns.value)

    def _capture_waveforms(self, frequency_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Capture simultaneous waveforms from channels A and B."""

        total_samples = self.samples_per_waveform
        capture_time_s = self.cycles_per_waveform / frequency_hz
        target_interval_ns = (capture_time_s / total_samples) * 1e9
        timebase, actual_interval_ns = self._find_timebase(total_samples, target_interval_ns)

        buffer_a_max = (ctypes.c_int16 * total_samples)()
        buffer_a_min = (ctypes.c_int16 * total_samples)()
        buffer_b_max = (ctypes.c_int16 * total_samples)()
        buffer_b_min = (ctypes.c_int16 * total_samples)()

        self.status["set_data_buffers_a"] = ps.ps3000aSetDataBuffers(
            self.chandle, 0, ctypes.byref(buffer_a_max), ctypes.byref(buffer_a_min), total_samples, 0, 0
        )
        assert_pico_ok(self.status["set_data_buffers_a"])
        self.status["set_data_buffers_b"] = ps.ps3000aSetDataBuffers(
            self.chandle, 1, ctypes.byref(buffer_b_max), ctypes.byref(buffer_b_min), total_samples, 0, 0
        )
        assert_pico_ok(self.status["set_data_buffers_b"])

        self.status["run_block"] = ps.ps3000aRunBlock(
            self.chandle, 0, total_samples, timebase, 1, None, 0, None, None
        )
        assert_pico_ok(self.status["run_block"])

        ready = ctypes.c_int16(0)
        while ready.value == 0:
            self.status["is_ready"] = ps.ps3000aIsReady(self.chandle, ctypes.byref(ready))

        overflow = (ctypes.c_int16 * 10)()
        captured = ctypes.c_int32(total_samples)
        self.status["get_values"] = ps.ps3000aGetValues(
            self.chandle, 0, ctypes.byref(captured), 0, 0, 0, ctypes.byref(overflow)
        )
        assert_pico_ok(self.status["get_values"])

        actual_samples = captured.value
        if any(val != 0 for val in overflow):
            logging.warning("ADC overflow detected at %.2f kHz", frequency_hz / 1000.0)

        # Convert the ADC counts to volts
        volts_a = np.array(adc2mV(buffer_a_max, self.channel_range, self.max_adc), dtype=float)
        volts_b = np.array(adc2mV(buffer_b_max, self.channel_range, self.max_adc), dtype=float)
        volts_a = volts_a[:actual_samples] * 1e-3
        volts_b = volts_b[:actual_samples] * 1e-3

        time_axis = np.arange(actual_samples, dtype=float) * (actual_interval_ns * 1e-9)

        self.status["stop"] = ps.ps3000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])

        return time_axis, volts_a, volts_b

    @staticmethod
    def _compute_complex_voltage(time_s: np.ndarray, signal_v: np.ndarray, frequency_hz: float) -> complex:
        """Return the complex amplitude (peak) of the signal at the target frequency."""

        omega = 2.0 * math.pi * frequency_hz
        sin_term = np.sin(omega * time_s)
        cos_term = np.cos(omega * time_s)
        ones = np.ones_like(time_s)
        design = np.column_stack((sin_term, cos_term, ones))
        coeffs, _, _, _ = np.linalg.lstsq(design, signal_v, rcond=None)
        sin_coeff, cos_coeff, _ = coeffs
        amplitude = math.hypot(sin_coeff, cos_coeff)
        phase = math.atan2(cos_coeff, sin_coeff)
        return complex(amplitude * math.cos(phase), amplitude * math.sin(phase))

    def _calculate_impedance(
        self,
        frequency_hz: float,
        time_s: np.ndarray,
        reference_signal: np.ndarray,
        dut_signal: np.ndarray,
    ) -> ImpedancePoint:
        reference_voltage = self._compute_complex_voltage(time_s, reference_signal, frequency_hz)
        dut_voltage = self._compute_complex_voltage(time_s, dut_signal, frequency_hz)

        if abs(reference_voltage) < 1e-12:
            raise RuntimeError("Reference voltage magnitude too small for impedance calculation.")

        impedance = self.reference_resistance * (dut_voltage / reference_voltage)
        return ImpedancePoint(frequency_hz, reference_voltage, dut_voltage, impedance, self.reference_resistance)

    def measure_frequency(self, frequency_hz: float) -> ImpedancePoint:
        """Measure impedance at a single frequency."""

        time_axis, ref_signal, dut_signal = self._capture_waveforms(frequency_hz)
        return self._calculate_impedance(frequency_hz, time_axis, ref_signal, dut_signal)

    def run(self) -> List[ImpedancePoint]:
        """Execute the impedance sweep."""

        self.open()
        results: List[ImpedancePoint] = []
        total = len(self.frequencies)
        try:
            for idx, frequency in enumerate(self.frequencies, start=1):
                self._configure_signal_generator(frequency)
                time.sleep(self.settle_time_s)
                point = self.measure_frequency(frequency)
                results.append(point)
                logging.info(
                    "Point %d/%d: %.3f kHz |Z|=%.3f Ω phase=%.2f°",
                    idx,
                    total,
                    frequency / 1000.0,
                    point.impedance_magnitude,
                    point.impedance_phase_deg,
                )
        finally:
            try:
                self._configure_signal_generator(max(self.freq_start_hz, 1.0), pk_to_pk_voltage=0.0)
            except PicoSDKCtypesError:
                logging.debug("Failed to disable signal generator during cleanup", exc_info=True)
            self.close()

        return results

    def close(self) -> None:
        """Close the PicoScope session."""

        try:
            status_code = ps.ps3000aCloseUnit(self.chandle)
        except PicoSDKCtypesError:
            logging.debug("Error closing PicoScope device", exc_info=True)
            return

        if status_code != 0:
            logging.debug("CloseUnit returned status %s", status_code)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def save_to_csv(self, results: Sequence[ImpedancePoint], output_path: Path) -> Path:
        """Persist the sweep to CSV and return the written path."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "frequency_hz",
            "frequency_khz",
            "impedance_real_ohms",
            "impedance_imag_ohms",
            "impedance_magnitude_ohms",
            "impedance_phase_deg",
            "current_amps",
            "voltage_ref_magnitude_volts",
            "voltage_ref_phase_deg",
            "voltage_dut_magnitude_volts",
            "voltage_dut_phase_deg",
        ]

        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for point in results:
                row = point.to_csv_row()
                row["frequency_khz"] = point.frequency_hz / 1000.0
                writer.writerow(row)

        logging.info("Saved %d impedance points to %s", len(results), output_path)
        return output_path

    @staticmethod
    def plot_results(results: Sequence[ImpedancePoint]) -> None:
        """Display a bode plot for the measured impedance."""

        if not results:
            logging.warning("No results to plot.")
            return

        freqs = np.array([point.frequency_hz for point in results])
        magnitudes = np.array([point.impedance_magnitude for point in results])
        phases = np.array([point.impedance_phase_deg for point in results])

        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

        ax_mag.set_xscale("log")
        ax_mag.plot(freqs, magnitudes, marker="", linestyle="-")
        ax_mag.set_ylabel("|Z| (Ω)")
        ax_mag.grid(True, which="both", linestyle=":")

        ax_phase.set_xscale("log")
        ax_phase.plot(freqs, phases, marker="", linestyle="-")
        ax_phase.set_ylabel("Phase (°)")
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.grid(True, which="both", linestyle=":")

        fig.tight_layout()
        plt.show()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PicoScope based impedance analyser")
    parser.add_argument("--start", type=float, default=30_000.0, help="Start frequency in Hz.")
    parser.add_argument("--stop", type=float, default=60_000.0, help="Stop frequency in Hz.")
    parser.add_argument(
        "--points-per-decade",
        type=int,
        default=2000,
        help="Number of sweep points per decade (default: 2000).",
    )
    parser.add_argument(
        "--vpp",
        type=float,
        default=2.0,
        help="Peak-to-peak output voltage for the stimulus in volts (default: 2 Vpp).",
    )
    parser.add_argument(
        "--reference",
        type=float,
        default=1_000.0,
        help="Reference resistor value in ohms used for impedance calculations.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="Number of samples captured per waveform (default: 4096).",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=10.0,
        help="Number of stimulus cycles recorded per capture window (default: 10).",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.05,
        help="Settling time in seconds after changing the generator frequency.",
    )
    parser.add_argument(
        "--range",
        type=int,
        default=8,
        help="Input range index for both channels (default corresponds to ±10 V).",
    )
    parser.add_argument(
        "--timebase-max",
        type=int,
        default=500,
        help="Maximum timebase index considered when resolving the sampling interval.",
    )
    parser.add_argument("--csv", type=Path, help="Optional path to save the sweep results as CSV.")
    parser.add_argument("--plot", action="store_true", help="Display a bode plot once the sweep completes.")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING). Default is INFO.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    analyzer = PicoImpedanceAnalyzer(
        freq_start_hz=args.start,
        freq_stop_hz=args.stop,
        points_per_decade=args.points_per_decade,
        voltage_pk_pk=args.vpp,
        reference_resistance_ohms=args.reference,
        samples_per_waveform=args.samples,
        cycles_per_waveform=args.cycles,
        settle_time_s=args.settle,
        channel_range=args.range,
        timebase_search_limit=args.timebase_max,
    )

    try:
        results = analyzer.run()
    except Exception:
        logging.exception("Impedance sweep failed")
        raise

    if args.csv:
        analyzer.save_to_csv(results, args.csv)

    if args.plot:
        analyzer.plot_results(results)
    else:
        logging.info("Sweep complete: acquired %d impedance points.", len(results))


if __name__ == "__main__":
    main()

