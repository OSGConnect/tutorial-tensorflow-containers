#!/bin/bash

set -e

echo
echo "I'm running on" $(hostname -f)
echo "OSG site: $OSG_SITE_NAME"
echo

python3 test.py 2>&1


