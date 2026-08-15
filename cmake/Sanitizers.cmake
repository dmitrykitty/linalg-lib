function(linalg_enable_sanitizers target_name)
    if(NOT LINALG_ENABLE_ASAN AND NOT LINALG_ENABLE_UBSAN)
        return()
    endif()

    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "The configured sanitizers currently require GCC or Clang")
    endif()

    set(enabled_sanitizers)
    if(LINALG_ENABLE_ASAN)
        list(APPEND enabled_sanitizers address)
    endif()
    if(LINALG_ENABLE_UBSAN)
        list(APPEND enabled_sanitizers undefined)
    endif()
    list(JOIN enabled_sanitizers "," sanitizer_list)

    target_compile_options(
        ${target_name}
        PRIVATE
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all
            -fsanitize=${sanitizer_list}
    )
    target_link_options(
        ${target_name}
        PRIVATE
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all
            -fsanitize=${sanitizer_list}
    )
endfunction()

