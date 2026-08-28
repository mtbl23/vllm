verse_sha256() {
  local tool
  if tool=$(type -P sha256sum) && [[ -x $tool ]]; then
    "$tool" "$@"
    return
  fi
  if tool=$(type -P shasum) && [[ -x $tool ]]; then
    "$tool" -a 256 "$@"
    return
  fi
  echo "sha256sum or shasum is required for SHA-256 verification" >&2
  return 127
}
