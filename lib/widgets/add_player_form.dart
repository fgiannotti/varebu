import 'package:flutter/material.dart';
import 'package:varebu/repositories/player.dart';
import 'package:varebu/repositories/player_in_memory.dart';

import '../main.dart';
import '../models/player.dart';

class AddPlayerForm extends StatefulWidget {
  final VoidCallback notifySave;

  const AddPlayerForm({super.key, required this.notifySave});

  @override
  AddPlayerFormState createState() {
    return AddPlayerFormState();
  }
}

class AddPlayerFormState extends State<AddPlayerForm> {
  final _formKey = GlobalKey<FormState>();
  late PlayerRepository repo;

  final nameCtrl = TextEditingController();
  final attackCtrl = TextEditingController();
  final blockCtrl = TextEditingController();
  final defenseCtrl = TextEditingController();
  final receptionCtrl = TextEditingController();
  final serveCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    repo = getIt<PlayerRepository>();
  }

  @override
  Widget build(BuildContext context) {
    var submitButton = Container(
      padding: const EdgeInsets.fromLTRB(0, 16, 0, 0),
      constraints: BoxConstraints.tight(const Size(128, 64)),
      child: ElevatedButton(
        onPressed: () {
          // Validate returns true if the form is valid, or false otherwise.
          var isValid = _formKey.currentState?.validate();
          if (isValid != null && isValid) {
            var player = buildPlayer();
            repo.insert(player);
            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
              content: Text('Jugador agregado correctamente'),
            ));
            widget.notifySave();
          }
        },
        child: const Text('Submit'),
      ),
    );
    return Form(
      key: _formKey,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Container(
            padding: const EdgeInsets.fromLTRB(4, 16, 4, 16),
            //constraints: BoxConstraints.tight(Size(128, 64)),
            child: Row(
              children: [
                TextForm('nombre', nameCtrl),
                TextForm('ataque', attackCtrl, onlyNumbers: true),
                TextForm('bloqueo', blockCtrl, onlyNumbers: true),
                TextForm('defensa', defenseCtrl, onlyNumbers: true),
                TextForm('recepcion', receptionCtrl, onlyNumbers: true),
                TextForm('saque', serveCtrl, onlyNumbers: true),
              ],
            ),
          ),
          submitButton,
        ],
      ),
    );
  }

  Expanded TextForm(String fieldName, TextEditingController controller,
      {onlyNumbers: false}) {
    var deco = InputDecoration(
      //hintText: 'What do people call you?',
      labelText: fieldName,
      labelStyle: const TextStyle(fontSize: 16),
    );
    var expanded = Expanded(
        child: Container(
            padding: const EdgeInsets.all(4),
            decoration: const BoxDecoration(
              border: Border(
                  top: BorderSide(), left: BorderSide(), bottom: BorderSide()),
            ),
            child: TextFormField(
              decoration: deco,
              controller: controller,
              keyboardType:
                  onlyNumbers ? TextInputType.number : TextInputType.name,
              validator: (value) {
                if (value == null ||
                    value.isEmpty ||
                    (onlyNumbers && int.tryParse(value) == null)) {
                  return 'Ingrese valor $fieldName';
                }
                return null;
              },
            )));
    return expanded;
  }

  Player buildPlayer() {
    int sum = int.parse(attackCtrl.text) +
        int.parse(blockCtrl.text) +
        int.parse(defenseCtrl.text) +
        int.parse(receptionCtrl.text) +
        int.parse(serveCtrl.text);
    return Player(nameCtrl.text, sum, attackCtrl.text, blockCtrl.text,
        defenseCtrl.text, receptionCtrl.text, serveCtrl.text);
  }
}
