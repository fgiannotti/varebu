import 'package:flutter/material.dart';
import 'package:varebu/repositories/player.dart';
import 'package:varebu/repositories/player_in_memory.dart';
import 'package:varebu/widgets/add_player_form.dart';
import 'package:varebu/widgets/buttons_row.dart';
import 'package:varebu/widgets/players_table.dart';
import 'package:get_it/get_it.dart';

import 'models/player.dart';

final getIt = GetIt.instance;

void main() {
  registerSingletons();
  runApp(const MyApp());
}

void registerSingletons() {
  getIt.registerSingleton<PlayerRepository>(InMemoryPlayerRepository());
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
      home: Home(),
    );
  }
}

class Home extends StatefulWidget {
  Home({super.key});
  List<Player> players = [];
  final repo = getIt<PlayerRepository>();

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
  bool rebuild = false;

  @override
  void initState() {
    widget.repo.insert(Player('Yulse', 300, '100', '40', '85', '85', '80'));
    widget.repo.insert(Player('Yulse', 300, '100', '40', '85', '85', '80'));
    widget.repo.insert(Player('Rakki', 300, '80', '90', '10', '15', '20'));
    widget.repo.insert(Player('Rakki', 300, '80', '90', '10', '15', '20'));
    widget.repo.insert(Player('Yulse', 300, '100', '40', '85', '85', '80'));
    print('INSERTED INITIAL PLAYERS');

    widget.repo.getAll().then((plyrs) {
      List<Player> result = [];
      for (var player in plyrs) {
        result.add(player);
      }

      setState(() {
        print('SETTED INITIAL STATE OF PLAYERS ${result.length}');
        widget.players = result;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    print('[Home] building with players ${widget.players.length}');
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
            PlayersTable(playerGetter: () { print('[MainGetter] returning ${widget.players.length}'); return widget.players; }),
            AddPlayerForm(
              notifySave: () {
                List<Player> result = [];
                widget.repo.getAll().then((plyrs) {
                  for (var player in plyrs) {
                    result.add(player);
                  }
                  setState(() {
                    rebuild = !rebuild;
                    widget.players = result;
                  });
                });

              },
            ),
          ],
        ),
      ),
    );
  }
}
