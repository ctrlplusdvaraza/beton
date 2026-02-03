#!/bin/sh

if [ ! -d .git ]; then
    echo "Error: This is not a git repository" >&2
    exit 1
fi

mkdir -p .git/hooks

cat > .git/hooks/commit-msg << 'EOF'
#!/bin/bash

# Regex to validate the type pattern
REGEX="^((Merge[ a-z-]* branch.*)|(Revert*)|((build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\(.*\))?!?: .*))"

FILE=`cat $1` # File containing the commit message

echo "Commit Message: ${FILE}"

if ! [[ $FILE =~ $REGEX ]]; then
	echo >&2 "ERROR: Commit aborted for not following the Conventional Commit standard.​"
	exit 1
else
	echo >&2 "Valid commit message."
fi
EOF

chmod +x .git/hooks/commit-msg

echo "Commit-msg hook has been installed successfully for this repository"

