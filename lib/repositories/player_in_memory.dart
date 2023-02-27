import 'dart:collection';

import 'package:varebu/repositories/player.dart';

import '../models/player.dart';

class Singleton {


}

class InMemoryPlayerRepository extends PlayerRepository {
  static final InMemoryPlayerRepository _singleton = InMemoryPlayerRepository._internal();

  factory InMemoryPlayerRepository() {
    return _singleton;
  }


  final Map<int, Player> db = HashMap();

  @override
  Future<List<Player>> getAll() {
    return Future.delayed(
        const Duration(seconds: 1), () => List.unmodifiable(db.values));
  }

  @override
  Future<Player?> getOne(int id) {
    return Future.delayed(
        const Duration(seconds: 1), () => db[id]);
  }

  @override
  Future<int> insert(Player player) {
    var id = db.values.length + 1;
    player.id = id;
    db.putIfAbsent(id, () => player);
    print('[repo] inserted in repo $id');
    return Future.delayed(
        const Duration(seconds: 1), () => player.id!);
  }

  @override
  Future<void> update(Player player) async {
    db.update(player.id!, (_) => player);
  }

  @override
  Future<void> delete(int id) async {
    db.remove(id);
  }

  InMemoryPlayerRepository._internal();
}
