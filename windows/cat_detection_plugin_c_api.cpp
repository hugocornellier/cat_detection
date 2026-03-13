#include "include/cat_detection/cat_detection_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "cat_detection_plugin.h"

void CatDetectionPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  cat_detection::CatDetectionPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
