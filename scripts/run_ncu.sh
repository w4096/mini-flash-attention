ncu \
  --config-file off \
  --export ./mfa \
  --force-overwrite \
  --target-processes application-only \
  --kernel-name "regex:flash|attention" \
  --warp-sampling-interval 1 \
  --warp-sampling-max-passes 1000 \
  --warp-sampling-buffer-size 536870912 \
  --set full \
  python benchmark/run.py
