file(READ "${INPUT}" CONTENT)

set(TAG "DELIMITERCL")

file(WRITE  "${OUTPUT}" "#pragma once\n#include <string>\n")
file(APPEND "${OUTPUT}" "const std::string ${VAR_NAME} = R\"${TAG}(\n")
file(APPEND "${OUTPUT}" "${CONTENT}")
file(APPEND "${OUTPUT}" "\n)${TAG}\";\n")
