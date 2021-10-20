# Writes CMAKE_CUDA_ARCHITECTURES to out_var, but using escaped semicolons
# as delimiters
function(nvbench_escaped_cuda_arches out_var)
  set(tmp)
  set(first TRUE)
  foreach(arg IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if (NOT first)
      string(APPEND tmp "\;")
    endif()
    string(APPEND tmp "${arg}")
    set(first FALSE)
  endforeach()
  set(${out_var} "${tmp}" PARENT_SCOPE)
endfunction()
