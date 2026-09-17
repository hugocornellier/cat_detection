import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// Guards the package-local species mapping that backs the species gate.
///
/// The gate drops any detection whose species is neither 'cat' nor
/// 'wildcat', so the exact contents of these blocks decide what the package
/// returns. A silent edit here would change detection behaviour with no code
/// change, which is what these assertions are for.
void main() {
  late Map<String, dynamic> mapping;
  late Map<String, dynamic> species;
  late List<dynamic> names;

  setUpAll(() {
    mapping = jsonDecode(
      File('assets/models/species_mapping.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    species = mapping['species'] as Map<String, dynamic>;
    names = mapping['imagenet_names'] as List<dynamic>;
  });

  List<int> idsOf(String block) =>
      (species[block] as Map<String, dynamic>)['imagenet_ids']
          .cast<int>()
          .toList();

  group('species mapping', () {
    test('declares only the target, near-miss and unknown blocks', () {
      expect(species.keys.toSet(), {'cat', 'wildcat', 'unknown_animal'});
    });

    test('carries the full 1000-name ImageNet table for breed lookup', () {
      expect(names.length, 1000);
    });

    test('domestic block has the expected size', () {
      expect(idsOf('cat').length, 5);
    });

    test('near-miss block has the expected size', () {
      expect(idsOf('wildcat').length, 2);
    });

    test('blocks do not overlap', () {
      final target = idsOf('cat').toSet();
      final near = idsOf('wildcat').toSet();
      expect(target.intersection(near), isEmpty);
    });

    test('unknown_animal maps no ids, so it is a fall-through only', () {
      expect(idsOf('unknown_animal'), isEmpty);
    });

    test('every mapped id is a valid ImageNet index', () {
      for (final block in ['cat', 'wildcat']) {
        for (final id in idsOf(block)) {
          expect(id, inInclusiveRange(0, 999), reason: 'block $block');
        }
      }
    });

    test('excluded neighbours stay unmapped so the gate drops them', () {
      final mapped = {...idsOf('cat'), ...idsOf('wildcat')};
      // Chosen deliberately: the landmark model never saw these and, being a
      // regressor with no confidence output, would emit plausible-looking
      // landmarks for them with no signal that anything was wrong.
      for (final id in [288, 289, 290, 291, 292, 293]) {
        expect(mapped.contains(id), isFalse, reason: names[id] as String);
      }
    });
  });
}
