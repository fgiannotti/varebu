import 'package:flutter/material.dart';
import 'package:varebu/repositories/player.dart';

import '../main.dart';
import '../models/player.dart';

class PlayersTable extends StatefulWidget {
  const PlayersTable({super.key});

  @override
  State<PlayersTable> createState() => _PlayersTableState();
}
// Row is composed by:
// 1. Player number
// 2. Player name
// 3. Player Stat SUM
// 3. Player Stat A (attack)
// 4. Player Stat B (Block)
// 5. Player Stat D (Defense)
// 6. Player Stat R (Reception)
// 7. Player Stat S (Serve)
// 8. Edit button ?
// 9. Delete button ?

class _PlayersTableState extends State<PlayersTable> {
  late PlayerRepository repo;
  List<TableRow> rows = [];
  @override
  void initState() {
    super.initState();
    repo = getIt<PlayerRepository>();
    repo.insert(Player('Yulse', 300, '100', '40', '85', '85', '80'));
    repo.insert(Player('Yulse', 300, '100', '40', '85', '85', '80'));
    repo.insert(Player('Rakki', 300, '80', '90', '10', '15', '20'));
    repo.insert(Player('Rakki', 300, '80', '90', '10', '15', '20'));
    repo.insert(Player('Yulse', 300, '100', '40', '85', '85', '80'));

    var playersFuture = repo.getAll();
    loadRows(playersFuture);
    print('initState done');
  }

  @override
  Widget build(BuildContext context) {
    print('building... rows: ${rows.length}');
    return Table(
      //border: TableBorder.symmetric(),
      columnWidths: const <int, TableColumnWidth>{
        0: FixedColumnWidth(4), // number
        1: FixedColumnWidth(64), // name
        2: FixedColumnWidth(8), // sum
        3: FixedColumnWidth(4), // A attack
        4: FixedColumnWidth(4), // B attack
        5: FixedColumnWidth(4), // D attack
        6: FixedColumnWidth(4), // R attack
        7: FixedColumnWidth(4), // S attack
        8: FixedColumnWidth(4), // button
      },
      children: <TableRow>[buildHeaderRow()] + rows,
    );
  }

  loadRows(Future<List<Player>> players) {
    List<TableRow> result = [];
    players.then(
            (plyrs) {
              for (var player in plyrs) {
                result.add(buildTableRow(player));
              }
              setState(() {
                print('setting state');
                rows = result;
              });
            });
  }

  TableRow buildTableRow(Player player) {
    var id = player.id!.toString();
    return TableRow(
        decoration: BoxDecoration(
            border: Border.all(strokeAlign: BorderSide.strokeAlignOutside),
            color: Colors.grey[300]),
        children: <Widget>[
          buildCell('$id.'),
          buildCell(player.name),
          buildCell(player.sum.toString()),
          buildCell(player.attack, isStatus: true),
          buildCell(player.block, isStatus: true),
          buildCell(player.defense, isStatus: true),
          buildCell(player.reception, isStatus: true),
          buildCell(player.serve, isStatus: true),
          IconButton(onPressed: () {}, icon: const Icon(Icons.edit_note)),
        ]);
  }

  TableCell buildCell(String text, {bool isStatus = false}) {
    return TableCell(
        verticalAlignment: TableCellVerticalAlignment.middle,
        child: Container(
          //padding: const EdgeInsets.fromLTRB(0, 0, 0, 0),
          color:
          isStatus ? fetchColorForStat(int.parse(text)) : Colors.grey[300],
          alignment: Alignment.center,
          child: Text(text),
        ));
  }

  buildHeaderRow() {
    return TableRow(children: <Widget>[
      const Text(''),
      buildCell(' Jugadores '),
      const Text(' sum ', textAlign: TextAlign.center),
      const Text('A', textAlign: TextAlign.center),
      const Text('B', textAlign: TextAlign.center),
      const Text('D', textAlign: TextAlign.center),
      const Text('R', textAlign: TextAlign.center),
      const Text('S', textAlign: TextAlign.center),
      const Text(''),
    ]);
  }

  fetchColorForStat(int stat) {
    if (stat == 100) {
      return Colors.greenAccent;
    }
    return Colors.green[(stat / 10).toInt() * 100];
  }

  Future<List<Player>> fetchPlayers() async {
    return await repo.getAll();
  }
}
