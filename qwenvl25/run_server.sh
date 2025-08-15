#!/bin/sh
#****************************************************************#
# ScriptName: run_server.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2025-04-17 10:33
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2025-04-17 10:33
# Function: 
#***************************************************************#
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python server.py
