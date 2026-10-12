# Live demo selection regression

Browser test against `python demo.py --device cpu` at http://127.0.0.1:8765.

1. Select example 3 / JPEG / Severe and run. Confirm source test:4720, airplane, approximately 99.02% calibrated confidence; all four policy cards populated.
2. Select example 1 without running. Assert the image is hidden, identity/cards empty, and status requests a new run. Repeat a condition and severity change after a completed run: results must clear.
3. Run again. During inference, assert the example/condition/severity controls are disabled. Once complete they unlock, except severity stays disabled for Clean.
4. Assert the result identifies test:5806 and the airplane's wrong ship prediction, approximately 99.87% calibrated confidence. The explanation and displayed result must now belong to the same source.

Observed RED on 12 October 2026: after changing from example 3 to 1, old test:4720/JPEG image, four cards and completion status remained beside the new explanation. GREEN after fix: selection clears prior results and controls cannot change during inference. The test reads DOM state and performs actual browser actions; it does not replace model responses.
