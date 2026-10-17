function(linalg_enable_sanitizers target_name)
    if(NOT LINALG_ENABLE_ASAN AND NOT LINALG_ENABLE_UBSAN AND NOT LINALG_ENABLE_TSAN)
        return()
    endif()

    if(LINALG_ENABLE_TSAN AND LINALG_ENABLE_ASAN)
        message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer")
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
    if(LINALG_ENABLE_TSAN)
        list(APPEND enabled_sanitizers thread)
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

