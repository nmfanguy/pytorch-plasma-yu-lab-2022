#include <torch/csrc/jit/mobile/import_data.h>

#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <torch/custom_class.h>

#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace jit {
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

namespace {

// The deserializer class which loads the bytecode package from bc files.
class BytecodeDeserializer final {
 public:
  explicit BytecodeDeserializer(std::unique_ptr<PyTorchStreamReader> reader);
  c10::IValue deserialize(c10::optional<at::Device> device);

 private:
  c10::IValue readArchive(
      const std::string& archive_name,
      std::shared_ptr<mobile::CompilationUnit> mcu,
      c10::optional<at::Device> device);

  std::shared_ptr<CompilationUnit> compilation_unit_;
  std::unique_ptr<PyTorchStreamReader> reader_;
};

BytecodeDeserializer::BytecodeDeserializer(
    std::unique_ptr<PyTorchStreamReader> reader)
    : compilation_unit_(std::make_shared<CompilationUnit>()),
      reader_(std::move(reader)) {}

c10::IValue BytecodeDeserializer::deserialize(
    c10::optional<at::Device> device) {
  auto mcu = std::make_shared<mobile::CompilationUnit>();

  // NOLINTNEXTLINE(performance-move-const-arg)
  return readArchive("data", mcu, std::move(device));
}

c10::IValue BytecodeDeserializer::readArchive(
    const std::string& archive_name,
    std::shared_ptr<mobile::CompilationUnit> mcu,
    c10::optional<at::Device> device) {
  std::stringstream picklename;
  picklename << archive_name << ".pkl";
  at::DataPtr pickle_ptr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) = reader_->getRecord(picklename.str());

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;
    }
    len = std::min(pickle_size - bytes_read, len);
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return len;
  };

  static const c10::QualifiedName torchPrefix = "__torch__";
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    TypePtr type;
    // HACK: first we check whether the name starts with `__torch__` to tell if
    // it's "supposed" to be a class type. This is a reliable check today, but
    // there is no guarantee that this is the case. The real solution is to
    // merge type parsers so we can share class resolution logic.
    if (torchPrefix.isPrefixOf(qn)) {
      if (compilation_unit_->get_class(qn) == nullptr) {
        auto typeptr = ClassType::create(qn, compilation_unit_, true);
        compilation_unit_->register_type(typeptr);
      }
      type = compilation_unit_->get_class(qn);
    } else {
      type = c10::parseType(qn.qualifiedName());
    }
    return c10::StrongTypePtr(compilation_unit_, type);
  };

  auto obj_loader = [&](const at::StrongTypePtr& type, IValue input) {
    auto cls = type.type_->expect<at::ClassType>();
    auto qn = cls->name();
    c10::QualifiedName method_name(qn.value(), "__setstate__");
    auto setstate = mcu->find_function(method_name);
    auto find_custom_class_with_setstate = [&qn]() -> c10::ClassTypePtr {
      auto custom_class_type = torch::jit::getCustomClass(qn->qualifiedName());
      if (custom_class_type && custom_class_type->findMethod("__setstate__")) {
        return custom_class_type;
      }
      return nullptr;
    };
    if (setstate) {
      auto obj = c10::ivalue::Object::create(type, 0);
      Stack stack({obj, input});
      setstate->run(stack);
      return obj;
    } else if (auto custom_class_type = find_custom_class_with_setstate()) {
      auto obj = c10::ivalue::Object::create(
          c10::StrongTypePtr(nullptr, custom_class_type), 1);
      Stack stack({obj, input});
      custom_class_type->getMethod("__setstate__").run(stack);
      return obj;
    } else {
      auto dict = std::move(input).toGenericDict();
      size_t ndict = dict.size();
      auto obj = c10::ivalue::Object::create(type, ndict);
      auto it = dict.begin();
      for (const auto i : c10::irange(ndict)) {
        std::stringstream name;
        name << it->key();
        cls->addOrCheckAttribute(name.str(), it->key().type());
        obj->setSlot(i, it->value());
        ++it;
      }
      return obj;
    }
  };

  auto read_record = [&](const std::string& name) {
    std::stringstream ss;
    ss << archive_name << "/" << name;
    return std::get<0>(reader_->getRecord(ss.str()));
  };

  Unpickler unpickler(
      reader,
      std::move(type_resolver),
      std::move(obj_loader),
      std::move(read_record),
      // NOLINTNEXTLINE(performance-move-const-arg)
      std::move(device),
      false,
      nullptr);
  return unpickler.parse_ivalue();
}

} // namespace

std::map<std::string, at::Tensor> _load_parameters(
    std::istream& in,
    c10::optional<at::Device> device) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  // NOLINTNEXTLINE(performance-move-const-arg)
  return _load_parameters(std::move(rai), std::move(device));
}

std::map<std::string, at::Tensor> _load_parameters(
    const std::string& filename,
    c10::optional<at::Device> device) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  // NOLINTNEXTLINE(performance-move-const-arg)
  return _load_parameters(std::move(rai), std::move(device));
}

std::map<std::string, at::Tensor> _load_parameters(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device) {
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  BytecodeDeserializer deserializer(std::move(reader));
  // NOLINTNEXTLINE(performance-move-const-arg)
  auto result = deserializer.deserialize(std::move(device)).toGenericDict();
  std::map<std::string, at::Tensor> map;
  for (const auto& e : result) {
    auto key = e.key().toString()->string();
    auto value = e.value().toTensor().tensor_data();
    map[key] = value;
  }
  return map;
}

} // namespace jit
} // namespace torch
