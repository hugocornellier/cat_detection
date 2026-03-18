//
//  Generated file. Do not edit.
//

// clang-format off

#include "generated_plugin_registrant.h"

#include <cat_detection/cat_detection_plugin.h>
#include <file_selector_linux/file_selector_plugin.h>

void fl_register_plugins(FlPluginRegistry* registry) {
  g_autoptr(FlPluginRegistrar) cat_detection_registrar =
      fl_plugin_registry_get_registrar_for_plugin(registry, "CatDetectionPlugin");
  cat_detection_plugin_register_with_registrar(cat_detection_registrar);
  g_autoptr(FlPluginRegistrar) file_selector_linux_registrar =
      fl_plugin_registry_get_registrar_for_plugin(registry, "FileSelectorPlugin");
  file_selector_plugin_register_with_registrar(file_selector_linux_registrar);
}
