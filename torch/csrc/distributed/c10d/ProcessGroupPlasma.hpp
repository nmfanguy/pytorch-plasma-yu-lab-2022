#pragma once

#include <chrono>
#include <condition_variable>
#include <string>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

namespace c10d {

const char* PLASMA_BACKEND_NAME = "plasma";

class TORCH_API ProcessGroupPlasma : public ProcessGroup {
  public:

    struct TORCH_API Options : public ProcessGroup::Options {
      explicit Options(
        std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout);

      static c10::intrusive_ptr<Options> create(
          std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout) {
        return c10::make_intrusive<Options>(timeout);
      }

      // this is pulled from the Gloo backend implementation, not sure if it will be
      // required for a Plasma version of the backend
      // std::vector<std::shared_ptr<::plasma::DEVICE>> devices;
      int threads;
    };

    const std::string getBackendName() const override {
      return std::string(PLASMA_BACKEND_NAME);
    }

    explicit ProcessGroupPlasma(
        const c10::intrusive_ptr<Store>& store,
        int rank,
        int size,
        c10::intrusive_ptr<Options> options = Options::create());

    virtual ~ProcessGroupPlasma();

    c10::intrusive_ptr<Options> getOptions() {
      return options_;
    } 

    c10::intrusive_ptr<ProcessGroup::Work> broadcast(
        std::vector<at::Tensor>& tensors,
        const BroadcastOptions& opts = BroadcastOptions()) override;
};
} // namespace c10d