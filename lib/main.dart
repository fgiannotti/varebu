import 'package:flutter/material.dart';
import 'package:varebu/widgets/buttons_row.dart';

import 'models/player.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Varebu',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // Try running your application with "flutter run". You'll see the
        // application has a blue toolbar. Then, without quitting the app, try
        // changing the primarySwatch below to Colors.green and then invoke
        // "hot reload" (press "r" in the console where you ran "flutter run",
        // or simply save your changes to "hot reload" in a Flutter IDE).
        // Notice that the counter didn't reset back to zero; the application
        // is not restarted.
        primaryColor: Colors.indigo,
        primarySwatch: Colors.indigo,
        colorScheme: const ColorScheme.light(
            primary: Colors.indigo, secondary: Colors.amber),
      ),
      home: const Home(),
    );
  }
}

class Home extends StatefulWidget {
  const Home({super.key});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: ListView(
          children: <Widget>[
            ButtonsRow(),
            Container(
                //constraints: BoxConstraints(maxWidth:32, maxHeight:60,minWidth:32,minHeight:32,),
                padding: const EdgeInsets.fromLTRB(32, 8, 32, 8),
                child: ElevatedButton(
                    child: Text('Agregar Jugador'),
                    onPressed: () {},
                ),
            ),
            PlayersTable(),
          ],
        ),
      ),
    );
  }
}

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
  @override
  Widget build(BuildContext context) {
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
      children: <TableRow>[
        buildHeaderRow(),
        buildTableRow(
            const Player('Yulse', '300', '100', '40', '85', '85', '80'), 1),
        buildTableRow(
            const Player('Rakki', '300', '80', '90', '10', '15', '20'), 1),
        buildTableRow(
            const Player('Yulse', '300', '95', '40', '85', '85', '80'), 1),
        buildTableRow(
            const Player('Yulse', '300', '95', '40', '85', '85', '80'), 1),
        buildTableRow(
            const Player('Yulse', '300', '95', '40', '85', '85', '80'), 1),
        buildTableRow(
            const Player('Yulse', '300', '95', '40', '85', '85', '80'), 1),
        buildTableRow(
            const Player('Yulse', '300', '95', '40', '85', '85', '80'), 1),
        buildTableRow(
            const Player('Yulse', '300', '95', '40', '85', '85', '80'), 1),
      ],
    );
  }

  TableRow buildTableRow(Player player, int i) {
    return TableRow(
        decoration: BoxDecoration(
            border: Border.all(strokeAlign: BorderSide.strokeAlignOutside),
            color: Colors.grey[300]),
        children: <Widget>[
          buildCell(i.toString() + '.'),
          buildCell(player.name),
          buildCell(player.sum),
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
    return Colors.green[(stat / 10).toInt() * 100];
  }
}
