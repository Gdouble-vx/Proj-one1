#!/bin/bash
PW="12345678"
echo "Checking sshpass..."
if which sshpass > /dev/null 2>&1; then
  echo "sshpass already installed"
else
  echo "Installing sshpass..."
  echo $PW | sudo -S apt-get install -y sshpass 2>&1 | tail -5
  echo "Done"
fi
