#!/bin/sh
SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cp ${SCRIPT_LOCATION}/hooks/* ${SCRIPT_LOCATION}/.git/hooks
