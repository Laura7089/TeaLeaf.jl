#!/bin/env -S julia --project
using TeaLeaf, Logging

settings = Settings()
chunk = initialiseapp!(settings)
@info "Solution Parameters" settings
diffuse!(chunk, settings)
