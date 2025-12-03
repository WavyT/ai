#!/usr/bin/env python3
"""
Test script to launch GUI, load data, and debug issues.
"""

import sys
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from advanced_eeg_gui2 import AdvancedEEGGUI

def test_gui():
    """Launch GUI and test functionality."""
    app = QApplication(sys.argv)

    # Create window
    window = AdvancedEEGGUI()
    window.show()

    print("=" * 60)
    print("GUI LAUNCHED")
    print("=" * 60)

    # Wait for GUI to initialize
    QApplication.processEvents()
    time.sleep(0.5)

    # Check if data was auto-loaded
    if window.loader is not None:
        print(f"✓ Data auto-loaded: {window.loader.num_channels} channels")

        # Select 2 channels for testing (12 and 67)
        print("\nSelecting channels 12 and 67 for testing...")
        window.channel_list.item(12).setSelected(True)
        window.channel_list.item(67).setSelected(True)
        window.on_channel_selection_changed()

        QApplication.processEvents()
        time.sleep(0.5)

        print(f"✓ Selected channels: {window.selected_channels}")
        print(f"✓ Current data shape: {window.current_data.shape if window.current_data is not None else 'None'}")

        # Check channel spacing
        print(f"\nChannel spacing: {window.channel_spacing:.2f}")
        print(f"Base Y scale: {window.base_y_scale:.2f}")

        # Check plot curves
        print(f"Plot curves created: {len(window.plot_curves)}")
        for ch_idx, curve in window.plot_curves.items():
            data = curve.getData()
            if data[0] is not None and data[1] is not None:
                y_data = data[1]
                print(f"  Ch {ch_idx}: y_range = [{y_data.min():.2f}, {y_data.max():.2f}]")

        # Test trigger detection
        print("\n" + "=" * 60)
        print("Testing Trigger Detection...")
        print("=" * 60)

        window.trigger_channel_spin.setValue(67)
        window.trigger_threshold_spin.setValue(1.2)  # Above our pulse amplitude of 2.0
        window.refractory_spin.setValue(21.0)

        print("Detecting triggers...")
        window.detect_and_plot_triggers()

        QApplication.processEvents()
        time.sleep(0.5)

        print(f"✓ Triggers detected: {len(window.detected_triggers)}")
        if len(window.detected_triggers) > 0:
            trigger_times = [t / window.sampling_rate for t in window.detected_triggers]
            print(f"  First 5 trigger times: {trigger_times[:5]}")

        # Check overview
        print("\n" + "=" * 60)
        print("Checking Overview Widget...")
        print("=" * 60)

        if window.overview_loaded:
            print(f"✓ Overview loaded with {len(window.overview_data)} points")
            print(f"  Overview data range: [{window.overview_data.min():.2f}, {window.overview_data.max():.2f}]")

            # Check overview Y range
            vb = window.overview_widget.getViewBox()
            if vb:
                y_range = vb.viewRange()[1]
                print(f"  Overview Y-axis range: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
        else:
            print("✗ Overview not loaded")

        print("\n" + "=" * 60)
        print("ISSUES FOUND:")
        print("=" * 60)

        # Check for overlapping channels
        if len(window.plot_curves) >= 2:
            curves_data = []
            for ch_idx in window.selected_channels:
                if ch_idx in window.plot_curves:
                    curve = window.plot_curves[ch_idx]
                    data = curve.getData()
                    if data[1] is not None:
                        curves_data.append((ch_idx, data[1].min(), data[1].max(), data[1].mean()))

            if len(curves_data) >= 2:
                # Check if ranges overlap
                for i in range(len(curves_data) - 1):
                    ch1, min1, max1, mean1 = curves_data[i]
                    ch2, min2, max2, mean2 = curves_data[i + 1]

                    overlap = not (max1 < min2 or max2 < min1)
                    separation = abs(mean2 - mean1)

                    print(f"\nCh {ch1} vs Ch {ch2}:")
                    print(f"  Ch {ch1}: range=[{min1:.2f}, {max1:.2f}], mean={mean1:.2f}")
                    print(f"  Ch {ch2}: range=[{min2:.2f}, {max2:.2f}], mean={mean2:.2f}")
                    print(f"  Separation: {separation:.2f}")
                    print(f"  Overlapping: {'YES ✗' if overlap else 'NO ✓'}")
                    print(f"  Expected spacing: {window.channel_spacing:.2f}")

    else:
        print("✗ No data loaded")

    # Schedule exit
    print("\n" + "=" * 60)
    print("Test complete. Exiting...")
    print("=" * 60)

    QTimer.singleShot(1000, app.quit)

    return app.exec()

if __name__ == "__main__":
    sys.exit(test_gui())
