# file_to_string(file_in file_out string_name)
#
# Create a C++ file `file_out` that defines a string named `string_name` in
# `namespace`, which contains the contents of `file_in`.

function(file_to_string file_in file_out namespace string_name)
  file(READ "${file_in}" file_in_contents)

  set(file_out_contents)
  string(APPEND file_out_contents
    "#include <string>\n"
    "namespace ${namespace} {\n"
    "const std::string ${string_name} =\n"
    "R\"expected(${file_in_contents})expected\";\n"
    "}\n"
  )

  file(WRITE "${file_out}" "${file_out_contents}")
endfunction()
