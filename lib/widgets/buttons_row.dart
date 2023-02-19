import 'package:flutter/material.dart';

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
    widget.onPressed(!widget.active); }

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