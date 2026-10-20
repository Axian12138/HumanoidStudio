# SPDX-License-Identifier: Apache-2.0
"""Build the hurdle scene, step it on the available device, and print status."""

from scene import sim, tracker

sim.run(10).sync()
print(tracker.last)
