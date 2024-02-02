# Memory Address to Search

The memory address are found in this [Repository](https://github.com/pret/pokecrystal).

## Actual Search

#### Improve fight rewards

The idea of researching these battle flags is to add a reward for using different movements and encourage AI to fight

### Battle Reward - 2 bits

00:c686 wBattleReward

---

### Poke Move PP - 4 bits

<!-- 03:d122 wBT_OTMon1PP
03:d15d wBT_OTMon2PP
03:d198 wBT_OTMon3PP -->

- 00:c634 wBattleMonPP

- 01:d29f wOTPartyMon1PP
- 01:d2cf wOTPartyMon2PP
- 01:d2ff wOTPartyMon3PP
- 01:d32f wOTPartyMon4PP
- 01:d35f wOTPartyMon5PP
- 01:d38f wOTPartyMon6PP

- 01:dcf6 wPartyMon1PP
- 01:dd26 wPartyMon2PP
- 01:dd56 wPartyMon3PP
- 01:dd86 wPartyMon4PP
- 01:ddB6 wPartyMon5PP
- 01:ddE6 wPartyMon6PP

---

### Poke Moves - 4 bits

- 01:dce1 wPartyMon1Moves
- 01:dd11 wPartyMon2Moves
- 01:dd41 wPartyMon3Moves
- 01:dd71 wPartyMon4Moves
- 01:dda1 wPartyMon5Moves
- 01:ddd1 wPartyMon6Moves

### Battle Type

- 01:d22d wBattleMode
- 01:d230 wBattleType
