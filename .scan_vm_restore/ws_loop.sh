#!/bin/bash
cd /content
while true; do python3 /content/whisper_server.py >> /content/whisper_server.log 2>&1; sleep 3; done
