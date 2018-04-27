#!/usr/bin/env bash

versionFile='version/version.txt'

OLDV=`cat ${versionFile}`
IFS=. components=(${OLDV##*-})
major=${components[0]}
minor=${components[1]}
patch=${components[2]}

newpatch=$((patch+1))

NEWV="${major}.${minor}.${newpatch}"
echo "${NEWV}" > "${versionFile}"
echo "${NEWV}"
