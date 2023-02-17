import 'package:flutter/material.dart';

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

class ButtonsRow extends StatefulWidget {
  const ButtonsRow({super.key});

  @override
  State<ButtonsRow> createState() => _ButtonsRowState();
}

class _ButtonsRowState extends State<ButtonsRow> {
  bool _playersSelected = true;
  bool _teamsSelected = false;

  void _handlePlayersButtonTap(bool newValue) {
    setState(() {
      _playersSelected = true;
      _teamsSelected = false;
    });
  }

  void _handleTeamsButtonTap(bool newValue) {
    setState(() {
      _playersSelected = false;
      _teamsSelected = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      child: Row(
        children: [
          Expanded(
            child: TopButton(
                text: 'Jugadores',
                active: _playersSelected,
                onPressed: _handlePlayersButtonTap
            ),
          ),
          Expanded(
            child: TopButton(
                text: 'Equipos',
                active: _teamsSelected,
                onPressed: _handleTeamsButtonTap
            ),
          ),
        ],
      ),
    );
  }
}

class TopButton extends StatefulWidget {
  final String text;
  final ValueChanged<bool> onPressed;
  bool active = false;

  TopButton(
      {super.key,
      required this.active,
      required this.text,
      required this.onPressed});

  @override
  State<TopButton> createState() => _TopButtonState();
}

class _TopButtonState extends State<TopButton> {
  void _handleTap() {
    widget.onPressed(!widget.active);
  }

  Color textColor() { return widget.active ? Colors.white : Colors.indigo;}

  @override
  Widget build(BuildContext context) {
    var backgroundColor = MaterialStateProperty.resolveWith<Color>((Set<MaterialState> states) { return widget.active ? Colors.indigo : Colors.grey; });

    return SizedBox(
      // width: MediaQuery.of(context).size.width,
      height: 50,
      child: OutlinedButton(
        onPressed: _handleTap,
        style: ButtonStyle(
          backgroundColor: backgroundColor,
        ),
        child: Text(widget.text + widget.active.toString(), style: TextStyle(color: textColor())),
      ),
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
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      // This call to setState tells the Flutter framework that something has
      // changed in this State, which causes it to rerun the build method below
      // so that the display can reflect the updated values. If we changed
      // _counter without calling setState(), then the build method would not be
      // called again, and so nothing would appear to happen.
      _counter++;
    });
  }

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
            const Text(
              'You have pushed the button this many times:',
            ),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headline4,
            ),
          ],
        ),
      ),
    );
  }
}
