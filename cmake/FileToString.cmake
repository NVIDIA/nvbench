# file_to_string(file_in file_out string_name)
#
# Create a C++ file `file_out` that defines a string named `string_name` in
# `namespace`, which contains the contents of `file_in`.

# Cache this so we can access it from wherever file_to_string is called.
set(_nvbench_file_to_string_path "${CMAKE_CURRENT_LIST_DIR}/FileToString.in")
function(file_to_string file_in file_out namespace string_name)
  file(READ "${file_in}" file_in_contents)

  set(file_out_contents)
  string(APPEND file_to_string_payload
    "#include <string>\n"
    "namespace ${namespace} {\n"
    "const std::string ${string_name} =\n"
    "R\"expected(${file_in_contents})expected\";\n"
    "}\n"
  )

  configure_file("${_nvbench_file_to_string_path}" "${file_out}")
endfunction()
