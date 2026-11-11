#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://cloud.ilabt.imec.be/public.php/dav/files/eRkdk6NnJZNL5XG"
DEST_ROOT="${1:-/usr/local/yzy/keyan2/datasets/PARrad/raw}"
REQUEST_HEADER="X-Requested-With: XMLHttpRequest"

download_file() {
  local subject="$1"
  local set_name="$2"
  local file_name="$3"
  local url="${BASE_URL}/${subject}/captured_data/${set_name}/${file_name}"
  local destination_dir="${DEST_ROOT}/${subject}/captured_data/${set_name}"
  local destination="${destination_dir}/${file_name}"
  local partial="${destination}.part"
  local expected
  local actual

  mkdir -p "${destination_dir}"
  expected="$(curl --fail --silent --show-error --head \
    --header "${REQUEST_HEADER}" "${url}" \
    | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub("\\r", "", $2); print $2; exit}')"
  if [[ -z "${expected}" || ! "${expected}" =~ ^[0-9]+$ ]]; then
    echo "ERROR unable to obtain content length: ${url}" >&2
    return 1
  fi

  if [[ -f "${destination}" ]]; then
    actual="$(stat -c '%s' "${destination}")"
    if [[ "${actual}" == "${expected}" ]]; then
      echo "SKIP complete ${destination} bytes=${actual}"
      return 0
    fi
    echo "ERROR existing final file has wrong size: ${destination} expected=${expected} actual=${actual}" >&2
    return 1
  fi

  echo "START ${destination} expected_bytes=${expected}"
  curl --fail --location --show-error \
    --header "${REQUEST_HEADER}" \
    --continue-at - \
    --retry 50 \
    --retry-all-errors \
    --retry-delay 15 \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time 300 \
    --output "${partial}" \
    "${url}"

  actual="$(stat -c '%s' "${partial}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR downloaded size mismatch: ${partial} expected=${expected} actual=${actual}" >&2
    return 1
  fi
  mv "${partial}" "${destination}"
  echo "DONE ${destination} bytes=${actual}"
}

for subject_number in 02 03 04 05 06; do
  subject="subject_${subject_number}"
  if [[ "${subject_number}" == "06" ]]; then
    set_numbers=(000 001 002)
  else
    set_numbers=(000 001 002 003)
  fi
  for set_number in "${set_numbers[@]}"; do
    set_name="set${set_number}"
    download_file "${subject}" "${set_name}" "timestamp_speech.csv"
    download_file "${subject}" "${set_name}" "processed_data.h5"
  done
done

echo "ALL_DOWNLOADS_COMPLETE ${DEST_ROOT}"
