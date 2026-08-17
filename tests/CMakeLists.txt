add_executable(sd-public-api-test test_public_api.cpp)
target_link_libraries(sd-public-api-test PRIVATE stable-diffusion)
add_test(NAME sd-public-api COMMAND sd-public-api-test)
