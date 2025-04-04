#pragma once
#include "dynet/rnn-state-machine.h"
#include "dynet/tensor.h"
#include "wav.h"
#include "cnn.h"

#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

void load_model(std::vector<soundRealDataNoisy> data);