#ifndef FLUTTER_PLUGIN_CAT_DETECTION_PLUGIN_PRIVATE_H_
#define FLUTTER_PLUGIN_CAT_DETECTION_PLUGIN_PRIVATE_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace cat_detection {

class CatDetectionPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  CatDetectionPlugin();
  virtual ~CatDetectionPlugin();

  CatDetectionPlugin(const CatDetectionPlugin&) = delete;
  CatDetectionPlugin& operator=(const CatDetectionPlugin&) = delete;

 private:
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace cat_detection

#endif  // FLUTTER_PLUGIN_CAT_DETECTION_PLUGIN_PRIVATE_H_
