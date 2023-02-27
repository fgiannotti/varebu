import '../models/player.dart';

abstract class PlayerRepository {
  Future<List<Player>> getAll();
  Future<Player?> getOne(int id);
  Future<int> insert(Player player);
  Future<void> update(Player player);
  Future<void> delete(int id);
}