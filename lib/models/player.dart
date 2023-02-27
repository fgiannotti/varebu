class Player {
  int? id;
  final String name;
  final int sum;
  final String attack;
  final String block;
  final String defense;
  final String reception;
  final String serve;

  Player(this.name, this.sum, this.attack, this.block, this.defense,
      this.reception, this.serve,[this.id = null]);
}
