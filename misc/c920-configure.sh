#!/bin/bash
# disable auto focus and auto white balance
v4l2-ctl --set-ctrl=white_balance_temperature_auto=0
v4l2-ctl --set-ctrl=white_balance_temperature=4000
v4l2-ctl --set-ctrl=focus_auto=0
v4l2-ctl --set-ctrl=focus_absolute=0
