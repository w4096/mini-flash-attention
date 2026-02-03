ncu \
  --config-file off \
  --export ./mfa.ncu-rep \
  --force-overwrite \
  --target-processes application-only \
  --kernel-name "regex:device|flash" \
  --warp-sampling-interval 1 \
  --warp-sampling-max-passes 1000 \
  --warp-sampling-buffer-size 536870912 \
  --set full \
  python ./tests/test.py
