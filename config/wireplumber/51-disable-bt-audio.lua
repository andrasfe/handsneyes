-- handsneyes: this host emulates a Bluetooth HID mouse ("devmouse"), not a
-- speaker or headset. By default PipeWire registers A2DP/HFP endpoints for
-- every connected Bluetooth device, which makes the paired Mac see devmouse
-- as an audio device and silently route its system output here — so the
-- operator loses call audio ("others can hear me, I can't hear them").
--
-- Empty roles = no audio endpoints registered = no audio profiles advertised.
-- To re-enable audio (e.g. for the speech_listener accessibility experiment,
-- see docs/audio-accessibility-experiment.md) delete this file and restart
-- wireplumber.
bluez_monitor.properties = bluez_monitor.properties or {}
bluez_monitor.properties["bluez5.roles"] = "[ ]"
bluez_monitor.properties["bluez5.hfphsp-backend"] = "none"
