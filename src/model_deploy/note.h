#ifndef NOTE_H
#define NOTE_H

#define song_number 8
#define song_length 400
#define Taiko_length 400

int note[6][12] = {
    33, 35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62,
    65, 69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123,
    131, 139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247,
    262, 277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494,
    523, 554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988,
    1047, 1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976
};

extern char song_name[song_number][18] = {
    "Moonlight Part 1",
    "Moonlight Part 2",
    "Moonlight Part 3",
    "Moonlight Part 4",
    "Moonlight Part 5",
    "Moonlight Part 6",
    "Moonlight Part 7",
    "Moonlight Part 8"
};

extern int song[song_number][song_length] = {{
    // 1
    note[1][1], note[1][8], note[2][1], note[2][4], note[2][8], note[2][1], note[2][4], note[2][8],
    note[3][1], note[2][4], note[2][8], note[3][1], note[3][4], note[2][8], note[3][1], note[3][4],
    note[3][8], note[3][1], note[3][4], note[3][8], note[4][1], note[3][4], note[3][8], note[4][1],
    note[4][4], note[3][8], note[4][1], note[4][4], note[4][8], note[4][8],
    // 3
    note[1][0], note[1][8], note[2][0], note[2][3], note[2][8], note[2][0], note[2][3], note[2][8],
    note[3][0], note[2][3], note[2][8], note[3][0], note[3][3], note[2][8], note[3][0], note[3][3],
    note[3][8], note[3][0], note[3][3], note[3][8], note[4][0], note[3][3], note[3][8], note[4][0],
    note[4][3], note[3][8], note[4][0], note[4][3], note[4][8], note[4][8],
    // 5
    note[0][11], note[2][1], note[2][5], note[2][8], note[3][1], note[2][5], note[2][8], note[3][1],
    note[3][5], note[2][8], note[3][1], note[3][5], note[2][8], note[3][1], note[3][5], note[3][8],
    note[4][1], note[3][5], note[3][8], note[4][1], note[4][5], note[3][8], note[4][1], note[4][5],
    note[4][8], note[4][1], note[4][5], note[4][8], note[5][1], note[5][1],
    // 7
    note[0][8], note[1][1], note[1][6], note[1][9], note[3][1], note[3][1], note[3][6], note[3][8],
    note[4][1], note[4][1], note[4][6], note[4][8], note[5][1], note[5][1],
    note[0][8], note[1][1], note[1][4], note[1][7], note[3][1], note[3][1], note[3][4], note[3][7],
    note[4][1], note[4][1], note[4][4], note[4][7], note[5][1], note[5][1],
    // 9
    note[5][0], note[3][8], note[4][8], note[3][8], note[4][8], note[3][10], note[4][8],
    note[4][0], note[4][8], note[5][1], note[4][8], note[4][3], note[4][8], note[4][0], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][6], note[4][8], note[4][4], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][0], note[4][8], note[3][9], note[4][7],
    // 11
    note[3][8], note[5][0], note[3][8], note[4][8], note[3][8], note[4][8], note[3][10], note[4][8],
    note[4][0], note[4][8], note[5][1], note[4][8], note[4][3], note[4][8], note[4][0], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][6], note[4][8], note[4][4], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][0], note[4][8], note[3][9], note[4][7],
    // 13
    note[3][8], note[4][8], note[3][9], note[4][7], note[3][8], note[4][8], note[3][9], note[4][7],
    note[3][8], note[4][8], note[3][9], note[4][7], note[3][8], note[4][8], note[3][9], note[4][7],
    note[3][8], note[2][8],
    // 15
    note[1][1], note[1][8], note[2][1], note[2][4], note[2][8], note[2][1], note[2][4], note[2][8],
    note[3][1], note[2][4], note[2][8], note[3][1], note[3][4], note[2][8], note[3][1], note[3][4],
    note[3][8], note[3][1], note[3][4], note[3][8], note[4][1], note[3][4], note[3][8], note[4][1],
    note[4][4], note[4][4], note[4][8], note[5][1], note[5][4], note[5][4],
    // 17
    note[0][10], note[1][4], note[1][7], note[3][1], note[3][4], note[1][7], note[3][1], note[3][4],
    note[3][7], note[3][1], note[3][4], note[3][7], note[4][1], note[3][4], note[3][7], note[4][1],
    note[4][4], note[3][7], note[4][1], note[4][4], note[4][7], note[4][1], note[4][4], note[4][7],
    note[5][1], note[4][4], note[4][7], note[5][1], note[5][4], note[5][4],
    // 19
    note[0][7], note[1][3], note[2][10], note[3][1], note[3][3], note[2][10], note[3][1], note[3][3],
    note[3][10], note[3][1], note[3][3], note[3][10], note[4][1], note[3][3], note[3][10], note[4][1],
    note[4][3], note[3][10], note[4][1], note[4][3], note[4][10], note[4][1], note[4][3], note[4][10],
    note[5][1], note[4][10], note[4][3], note[4][1], note[4][10], note[4][3], note[4][1], note[3][10]
    }, {
    // 21
    note[3][11], note[2][3], note[1][11], note[2][3], note[4][3], note[2][3], note[1][11], note[2][3],
    note[1][8], note[2][3], note[1][11], note[2][3], note[3][11], note[2][3], note[1][11], note[3][8],
    note[3][8], note[2][3], note[2][1], note[2][3], note[3][7], note[2][3], note[2][1], note[2][3],
    note[1][10], note[2][3], note[3][7], note[2][3], note[4][3], note[2][3], note[3][7], note[2][3],
    // 23
    note[3][10], note[2][3], note[1][11], note[2][3], note[3][8], note[2][3], note[1][11], note[2][3],
    note[1][11], note[2][3], note[3][8], note[2][3], note[4][3], note[2][3], note[1][11], note[3][8],
    note[3][11], note[2][3], note[1][7], note[2][3], note[3][10], note[2][3], note[1][7], note[2][3],
    note[1][7], note[2][3], note[3][10], note[2][3], note[4][3], note[2][3], note[1][7], note[3][10],
    // 25
    note[3][11], note[2][3], note[4][3], note[2][3], note[1][8], note[2][3], note[4][3], note[2][3],
    note[1][8], note[2][3], note[4][3], note[2][3], note[3][11], note[2][3], note[3][8], note[2][3],
    note[3][8], note[2][3], note[3][7], note[2][3], note[1][10], note[2][3], note[3][7], note[2][3],
    note[1][10], note[2][3], note[3][7], note[2][3], note[4][3], note[2][3], note[3][7], note[2][3],
    // 27
    note[3][10], note[2][3], note[3][8], note[2][3], note[1][10], note[2][3], note[3][8], note[2][3],
    note[1][10], note[2][3], note[3][8], note[2][3], note[4][3], note[2][3], note[3][8], note[2][3],
    note[3][11], note[2][3], note[3][10], note[2][3], note[1][7], note[2][3], note[3][10], note[2][3],
    note[1][7], note[2][3], note[3][10], note[2][3], note[4][3], note[2][3], note[3][10], note[2][3],
    // 29
    note[4][0], note[2][3], note[1][8], note[2][3], note[1][6], note[2][3], note[1][8], note[2][3],
    note[4][1], note[2][1], note[1][8], note[2][1], note[1][4], note[2][1], note[1][8], note[2][1],
    note[3][10], note[3][11], note[3][8], note[3][10], note[3][11], note[1][11], note[1][6], note[1][11],
    note[1][3], note[1][11], note[1][6], note[1][11], note[1][3], note[1][11], note[1][6], note[1][11],
    // 31
    note[3][8], note[1][11], note[1][4], note[1][11], note[1][2], note[1][11], note[1][4], note[1][11],
    note[3][9], note[1][8], note[1][4], note[1][8], note[1][1], note[1][8], note[1][4], note[1][8],
    note[3][7], note[3][8], note[3][5], note[3][7], note[3][8], note[1][8], note[1][3], note[1][8],
    note[0][11], note[1][8], note[1][3], note[1][8], note[0][11], note[1][8], note[1][3], note[1][8],
    // 33
    note[3][9],
    note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[4][11],
    note[4][9], note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[4][11],
    note[4][9], note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[4][11],
    // 35
    note[4][9], note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[5][2],
    note[5][4], note[5][3], note[5][4], note[5][3], note[5][4], note[5][1], note[4][11], note[4][9],
    note[4][8], note[2][3], note[1][11], note[2][3], note[1][6], note[2][3], note[1][11], note[2][3],
    note[4][10], note[4][11], note[4][10], note[4][11], note[4][10], note[4][11], note[4][8], note[4][10],
    // 37
    note[4][8], 1, note[3][9],
    note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    // 39
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[4][2],
    note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[5][2], note[5][3],
    // 41
    note[5][4], note[2][8], note[2][4], note[2][8], note[2][1], note[2][8], note[2][4], note[2][8],
    note[4][8], note[2][11], note[2][5], note[2][11], note[2][2], note[2][11], note[2][5], note[2][11],
    note[4][11], note[2][11], note[2][8], note[2][11], note[2][3], note[2][11], note[2][8], note[2][11],
    note[3][7], note[2][3], note[2][2], note[2][3], note[2][2], note[2][3], note[2][2], note[2][3]
    }, {
    // 43
    note[3][8], note[3][11], note[3][11], note[3][11],
    note[3][11], note[3][11], note[3][10], note[3][8],
    note[3][7], note[4][3], note[4][3], note[4][3],
    note[4][3], note[4][3], note[4][3], note[4][3],
    // 45
    note[3][8], note[3][11], note[3][11], note[3][11],
    note[3][11], note[3][11], note[3][10], note[3][8],
    note[3][7], note[4][3], note[4][3], note[4][3],
    note[4][3], note[4][3], note[4][3], note[4][3],
    // 47
    note[4][3], note[4][3], note[4][3], note[3][11],
    note[2][1], note[4][4], note[4][4], note[4][1],
    note[2][3], note[4][3], note[4][3], note[3][11],
    note[2][3], note[4][3], note[4][3], note[3][10],
    // 49
    note[3][11], note[4][11], note[4][11], note[4][11],
    note[4][11], note[4][11], note[4][10], note[4][8],
    note[4][7], note[5][4], note[5][4], note[5][4],
    note[5][4], note[5][4], note[5][3], note[5][1],
    // 51
    note[4][11], note[4][11], note[4][11], note[4][11],
    note[4][11], note[4][11], note[4][10], note[4][8],
    note[4][7], note[5][4], note[5][4], note[5][4],
    note[5][4], note[5][4], note[5][3], note[5][1],
    // 53
    note[4][11], note[4][11], note[4][11], note[5][3],
    note[2][1], note[5][1], note[5][1], note[5][4],
    note[2][3], note[4][11], note[4][11], note[5][3],
    note[2][1], note[4][10], note[4][10], note[5][3],
    // 55
    note[1][11], note[4][11], note[4][11], note[5][3],
    note[2][1], note[4][9], note[4][9], note[5][1],
    note[2][3], note[4][8], note[4][8], note[4][11],
    note[2][3], note[4][7], note[4][7], note[4][10],
    // 57
    note[4][8], note[2][8], note[2][3], note[2][8], note[1][8], note[2][8], note[2][3], note[2][8],
    note[1][8], note[2][8], note[2][3], note[2][8], note[3][11], note[2][8], note[2][3], note[3][8],
    note[3][7], note[2][10], note[2][3], note[2][10], note[4][3], note[2][10], note[2][3], note[2][10],
    note[4][3], note[2][10], note[2][3], note[2][10], note[4][3], note[2][10], note[2][3], note[2][10],
    // 59
    note[4][3], note[2][8], note[2][3], note[2][8], note[1][8], note[2][8], note[2][3], note[2][8],
    note[1][8], note[2][8], note[2][3], note[2][8], note[4][11], note[2][8], note[2][3], note[4][8],
    note[4][7], note[2][10], note[2][3], note[2][10], note[5][3], note[2][10], note[2][3], note[2][10],
    note[5][3], note[2][10], note[2][3], note[2][10], note[5][3], note[2][10], note[2][3], note[2][10],
    // 61
    note[5][3], note[2][8], note[2][3], note[2][8], note[1][8], note[2][8], note[2][3], note[2][8],
    note[5][3], note[2][10], note[2][3], note[2][10], note[1][8], note[2][10], note[2][3], note[2][10],
    note[5][3], note[2][8], note[2][3], note[2][8], note[1][8], note[2][8], note[2][3], note[2][8],
    note[5][3], note[2][10], note[2][3], note[2][10], note[1][8], note[2][10], note[2][3], note[2][10],
    // 63
    note[2][11], note[3][8], note[3][3], note[3][8], note[2][11], note[3][8], note[3][3], note[3][8],
    note[2][11], note[3][8], note[3][3], note[3][8], note[2][11], note[3][8], note[3][3], note[3][8],
    note[3][0], note[3][6], note[3][3], note[3][6], note[3][0], note[3][6], note[3][3], note[3][6],
    note[3][0], note[3][6], note[3][3], note[3][6], note[3][0], note[3][6], note[3][3], note[3][6]
    }, {
    // 1
    note[3][4], note[1][8], note[2][1], note[2][4], note[2][8], note[2][1], note[2][4], note[2][8],
    note[3][1], note[2][4], note[2][8], note[3][1], note[3][4], note[2][8], note[3][1], note[3][4],
    note[3][8], note[3][1], note[3][4], note[3][8], note[4][1], note[3][4], note[3][8], note[4][1],
    note[4][4], note[3][8], note[4][1], note[4][4], note[4][8], note[4][8],
    // 3
    note[1][0], note[1][8], note[2][0], note[2][3], note[2][8], note[2][0], note[2][3], note[2][8],
    note[3][0], note[2][3], note[2][8], note[3][0], note[3][3], note[2][8], note[3][0], note[3][3],
    note[3][8], note[3][0], note[3][3], note[3][8], note[4][0], note[3][3], note[3][8], note[4][0],
    note[4][3], note[3][8], note[4][0], note[4][3], note[4][8], note[4][8],
    // 5
    note[0][11], note[2][1], note[2][5], note[2][8], note[3][1], note[2][5], note[2][8], note[3][1],
    note[3][5], note[2][8], note[3][1], note[3][5], note[2][8], note[3][1], note[3][5], note[3][8],
    note[4][1], note[3][5], note[3][8], note[4][1], note[4][5], note[3][8], note[4][1], note[4][5],
    note[4][8], note[4][1], note[4][5], note[4][8], note[5][1], note[5][1],
    // 7
    note[0][8], note[1][1], note[1][6], note[1][9], note[3][1], note[3][1], note[3][6], note[3][8],
    note[4][1], note[4][1], note[4][6], note[4][8], note[5][1], note[5][1],
    note[0][8], note[1][1], note[1][4], note[1][7], note[3][1], note[3][1], note[3][4], note[3][7],
    note[4][1], note[4][1], note[4][4], note[4][7], note[5][1], note[5][1],
    // 9
    note[5][0], note[3][8], note[4][8], note[3][8], note[4][8], note[3][10], note[4][8],
    note[4][0], note[4][8], note[5][1], note[4][8], note[4][3], note[4][8], note[4][0], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][6], note[4][8], note[4][4], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][0], note[4][8], note[3][9], note[4][7],
    // 11
    note[3][8], note[5][0], note[3][8], note[4][8], note[3][8], note[4][8], note[3][10], note[4][8],
    note[4][0], note[4][8], note[5][1], note[4][8], note[4][3], note[4][8], note[4][0], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][6], note[4][8], note[4][4], note[4][8],
    note[4][3], note[4][8], note[4][1], note[4][8], note[4][0], note[4][8], note[3][9], note[4][7],
    // 13
    note[3][8], note[4][8], note[3][9], note[4][7], note[3][8], note[4][8], note[3][9], note[4][7],
    note[3][8], note[4][8], note[3][9], note[4][7], note[3][8], note[4][8], note[3][9], note[4][7],
    note[3][8], note[2][8],
    // 15
    note[1][1], note[1][8], note[2][1], note[2][4], note[2][8], note[2][1], note[2][4], note[2][8],
    note[3][1], note[2][4], note[2][8], note[3][1], note[3][4], note[2][8], note[3][1], note[3][4],
    note[3][8], note[3][1], note[3][4], note[3][8], note[4][1], note[3][4], note[3][8], note[4][1],
    note[4][4], note[4][4], note[4][8], note[5][1], note[5][4], note[5][4],
    // 17
    note[0][10], note[1][4], note[1][7], note[3][1], note[3][4], note[1][7], note[3][1], note[3][4],
    note[3][7], note[3][1], note[3][4], note[3][7], note[4][1], note[3][4], note[3][7], note[4][1],
    note[4][4], note[3][7], note[4][1], note[4][4], note[4][7], note[4][1], note[4][4], note[4][7],
    note[5][1], note[4][4], note[4][7], note[5][1], note[5][4], note[5][4],
    // 19
    note[0][7], note[1][3], note[2][10], note[3][1], note[3][3], note[2][10], note[3][1], note[3][3],
    note[3][10], note[3][1], note[3][3], note[3][10], note[4][1], note[3][3], note[3][10], note[4][1],
    note[4][3], note[3][10], note[4][1], note[4][3], note[4][10], note[4][1], note[4][3], note[4][10],
    note[5][1], note[4][10], note[4][3], note[4][1], note[4][10], note[4][3], note[4][1], note[3][10]
    }, {
    // 21
    note[3][11], note[2][3], note[1][11], note[2][3], note[4][3], note[2][3], note[1][11], note[2][3],
    note[1][8], note[2][3], note[1][11], note[2][3], note[3][11], note[2][3], note[1][11], note[3][8],
    note[3][8], note[2][3], note[2][1], note[2][3], note[3][7], note[2][3], note[2][1], note[2][3],
    note[1][10], note[2][3], note[3][7], note[2][3], note[4][3], note[2][3], note[3][7], note[2][3],
    // 23
    note[3][10], note[2][3], note[1][11], note[2][3], note[3][8], note[2][3], note[1][11], note[2][3],
    note[1][11], note[2][3], note[3][8], note[2][3], note[4][3], note[2][3], note[1][11], note[3][8],
    note[3][11], note[2][3], note[1][7], note[2][3], note[3][10], note[2][3], note[1][7], note[2][3],
    note[1][7], note[2][3], note[3][10], note[2][3], note[4][3], note[2][3], note[1][7], note[3][10],
    // 25
    note[3][11], note[2][3], note[4][3], note[2][3], note[1][8], note[2][3], note[4][3], note[2][3],
    note[1][8], note[2][3], note[4][3], note[2][3], note[3][11], note[2][3], note[3][8], note[2][3],
    note[3][8], note[2][3], note[3][7], note[2][3], note[1][10], note[2][3], note[3][7], note[2][3],
    note[1][10], note[2][3], note[3][7], note[2][3], note[4][3], note[2][3], note[3][7], note[2][3],
    // 27
    note[3][10], note[2][3], note[3][8], note[2][3], note[1][10], note[2][3], note[3][8], note[2][3],
    note[1][10], note[2][3], note[3][8], note[2][3], note[4][3], note[2][3], note[3][8], note[2][3],
    note[3][11], note[2][3], note[3][10], note[2][3], note[1][7], note[2][3], note[3][10], note[2][3],
    note[1][7], note[2][3], note[3][10], note[2][3], note[4][3], note[2][3], note[3][10], note[2][3],
    // 29
    note[4][0], note[2][3], note[1][8], note[2][3], note[1][6], note[2][3], note[1][8], note[2][3],
    note[4][1], note[2][1], note[1][8], note[2][1], note[1][4], note[2][1], note[1][8], note[2][1],
    note[3][10], note[3][11], note[3][8], note[3][10], note[3][11], note[1][11], note[1][6], note[1][11],
    note[1][3], note[1][11], note[1][6], note[1][11], note[1][3], note[1][11], note[1][6], note[1][11],
    // 31
    note[3][8], note[1][11], note[1][4], note[1][11], note[1][2], note[1][11], note[1][4], note[1][11],
    note[3][9], note[1][8], note[1][4], note[1][8], note[1][1], note[1][8], note[1][4], note[1][8],
    note[3][7], note[3][8], note[3][5], note[3][7], note[3][8], note[1][8], note[1][3], note[1][8],
    note[0][11], note[1][8], note[1][3], note[1][8], note[0][11], note[1][8], note[1][3], note[1][8],
    // 33
    note[3][9],
    note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[4][11],
    note[4][9], note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[4][11],
    note[4][9], note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[4][11],
    // 35
    note[4][9], note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[5][2],
    note[5][4], note[5][3], note[5][4], note[5][3], note[5][4], note[5][1], note[4][11], note[4][9],
    note[4][8], note[2][3], note[1][11], note[2][3], note[1][6], note[2][3], note[1][11], note[2][3],
    note[4][10], note[4][11], note[4][10], note[4][11], note[4][10], note[4][11], note[4][8], note[4][10],
    // 37
    note[4][8], 1, note[3][9],
    note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    // 39
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[3][11],
    note[3][9], note[3][4], note[3][6], note[3][8], note[3][9], note[3][11], note[4][1], note[4][2],
    note[4][4], note[4][6], note[4][8], note[4][9], note[4][11], note[5][1], note[5][2], note[5][3],
    // 41
    note[5][4], note[2][8], note[2][4], note[2][8], note[2][1], note[2][8], note[2][4], note[2][8],
    note[4][8], note[2][11], note[2][5], note[2][11], note[2][2], note[2][11], note[2][5], note[2][11],
    note[4][11], note[2][11], note[2][8], note[2][11], note[2][3], note[2][11], note[2][8], note[2][11],
    note[3][7], note[2][3], note[2][2], note[2][3], note[2][2], note[2][3], note[2][2], note[2][3]
    }, {
    // 43
    note[3][8], note[3][11], note[3][11], note[3][11],
    note[3][11], note[3][11], note[3][10], note[3][8],
    note[3][7], note[4][3], note[4][3], note[4][3],
    note[4][3], note[4][3], note[4][3], note[4][3],
    // 45
    note[3][8], note[3][11], note[3][11], note[3][11],
    note[3][11], note[3][11], note[3][10], note[3][8],
    note[3][7], note[4][3], note[4][3], note[4][3],
    note[4][3], note[4][3], note[4][3], note[4][3],
    // 47
    note[4][3], note[4][3], note[4][3], note[3][11],
    note[2][1], note[4][4], note[4][4], note[4][1],
    note[2][3], note[4][3], note[4][3], note[3][11],
    note[2][3], note[4][3], note[4][3], note[3][10],
    // 49
    note[3][11], note[4][11], note[4][11], note[4][11],
    note[4][11], note[4][11], note[4][10], note[4][8],
    note[4][7], note[5][4], note[5][4], note[5][4],
    note[5][4], note[5][4], note[5][3], note[5][1],
    // 51
    note[4][11], note[4][11], note[4][11], note[4][11],
    note[4][11], note[4][11], note[4][10], note[4][8],
    note[4][7], note[5][4], note[5][4], note[5][4],
    note[5][4], note[5][4], note[5][3], note[5][1],
    // 53
    note[4][11], note[4][11], note[4][11], note[5][3],
    note[2][1], note[5][1], note[5][1], note[5][4],
    note[2][3], note[4][11], note[4][11], note[5][3],
    note[2][1], note[4][10], note[4][10], note[5][3],
    // 55
    note[1][11], note[4][11], note[4][11], note[5][3],
    note[2][1], note[4][9], note[4][9], note[5][1],
    note[2][3], note[4][8], note[4][8], note[4][11],
    note[2][3], note[4][7], note[4][7], note[4][10],
    // 57
    note[4][8], note[2][8], note[2][3], note[2][8], note[1][8], note[2][8], note[2][3], note[2][8],
    note[1][8], note[2][8], note[2][3], note[2][8], note[3][11], note[2][8], note[2][3], note[3][8],
    note[3][7], note[2][10], note[2][3], note[2][10], note[4][3], note[2][10], note[2][3], note[2][10],
    note[4][3], note[2][10], note[2][3], note[2][10], note[4][3], note[2][10], note[2][3], note[2][10],
    // 59
    note[4][3], note[2][8], note[2][3], note[2][8], note[1][8], note[2][8], note[2][3], note[2][8],
    note[1][8], note[2][8], note[2][3], note[2][8], note[4][11], note[2][8], note[2][3], note[4][8],
    note[4][7], note[2][10], note[2][3], note[2][10], note[5][3], note[2][10], note[2][3], note[2][10],
    note[5][3], note[2][10], note[2][3], note[2][10], note[5][3], note[2][10], note[2][3], note[2][10],
    // 61
    note[5][3], note[2][8], note[2][3], note[2][8], note[1][8], note[2][8], note[2][3], note[2][8],
    note[5][3], note[2][10], note[2][3], note[2][10], note[1][8], note[2][10], note[2][3], note[2][10],
    note[5][3], note[2][8], note[2][3], note[2][8], note[1][8], note[2][8], note[2][3], note[2][8],
    note[5][3], note[2][10], note[2][3], note[2][10], note[1][8], note[2][10], note[2][3], note[2][10],
    // 63
    note[2][11], note[3][8], note[3][3], note[3][8], note[2][11], note[3][8], note[3][3], note[3][8],
    note[2][11], note[3][8], note[3][3], note[3][8], note[2][11], note[3][8], note[3][3], note[3][8],
    note[3][0], note[3][6], note[3][3], note[3][6], note[3][0], note[3][6], note[3][3], note[3][6],
    note[3][0], note[3][6], note[3][3], note[3][6], note[3][0], note[3][6], note[3][3], note[3][6]
    }, {
    // 65
    note[3][5], note[2][1], note[2][5], note[2][8], note[3][1], note[2][5], note[2][8], note[3][1],
    note[3][5], note[2][8], note[3][1], note[3][5], note[3][8], note[3][1], note[3][5], note[3][8],
    note[4][1], note[3][5], note[3][8], note[4][1], note[4][5], note[3][8], note[4][1], note[4][5],
    note[4][8], note[4][1], note[4][5], note[4][8], note[5][1], note[5][1],
    // 67
    note[0][11], note[2][1], note[2][5], note[2][8], note[3][1], note[2][5], note[2][8], note[3][1],
    note[3][5], note[2][8], note[3][1], note[3][5], note[3][8], note[3][1], note[3][5], note[3][8],
    note[4][1], note[3][5], note[3][8], note[4][1], note[4][5], note[3][8], note[4][1], note[4][5],
    note[4][8], note[4][1], note[4][5], note[4][8], note[5][1], note[5][1],
    // 69
    note[0][9], note[2][1], note[2][6], note[2][9], note[3][1], note[3][1], note[3][6], note[3][9],
    note[4][1], note[4][1], note[4][6], note[4][9], note[5][1], note[5][1],
    note[0][5], note[2][1], note[2][8], note[2][11], note[3][1], note[3][1], note[3][8], note[3][11],
    note[4][1], note[4][1], note[4][8], note[4][11], note[5][1], note[5][1],
    // 71
    note[5][1], note[3][1], note[2][9], note[3][1], note[4][1], note[3][1], note[2][9], note[3][1],
    note[2][6], note[3][1], note[2][9], note[3][1], note[3][9], note[3][1], note[2][9], note[3][6],
    note[3][6], note[3][1], note[2][11], note[3][1], note[3][5], note[3][1], note[2][11], note[3][1],
    note[2][8], note[3][1], note[3][5], note[3][1], note[4][1], note[3][1], note[2][11], note[3][5],
    // 73
    note[3][8], note[3][1], note[2][9], note[3][1], note[3][6], note[3][1], note[2][9], note[3][1],
    note[2][9], note[3][1], note[3][6], note[3][1], note[4][1], note[3][1], note[2][9], note[3][6],
    note[3][9], note[3][1], note[2][5], note[3][1], note[3][8], note[3][1], note[2][5], note[3][1],
    note[2][5], note[3][1], note[3][8], note[3][1], note[4][1], note[3][1], note[2][5], note[3][8],
    // 75
    note[2][6], note[1][4], note[3][9], note[1][4], note[3][1], note[4][1], note[3][9], note[4][1],
    note[3][6], note[4][1], note[3][9], note[4][1], note[2][9], note[4][1], note[3][9], note[2][6],
    note[2][6], note[4][1], note[3][11], note[4][11], note[2][5], note[4][1], note[3][11], note[4][11],
    note[3][8], note[4][1], note[2][5], note[4][11], note[3][1], note[4][1], note[3][11], note[2][5],
    // 77
    note[2][8], note[4][1], note[3][9], note[4][1], note[2][6], note[4][1], note[3][9], note[4][1],
    note[3][6], note[4][1], note[2][6], note[4][1], note[3][1], note[4][1], note[3][9], note[2][6],
    note[2][6], note[4][2], note[2][6], note[4][2], note[3][2], note[4][2], note[3][9], note[2][6],
    note[2][6], note[4][2], note[2][6], note[4][2], note[3][2], note[4][2], note[4][0], note[2][6],
    // 79
    note[2][7], note[4][2], note[3][11], note[4][2], note[2][2], note[4][2], note[3][11], note[4][2],
    note[3][7], note[4][2], note[3][11], note[4][2], note[1][11], note[4][2], note[3][11], note[1][7],
    note[1][7], note[4][2], note[4][0], note[4][2], note[1][6], note[4][2], note[4][0], note[4][2],
    note[3][9], note[4][2], note[1][6], note[4][2], note[2][2], note[4][2], note[4][0], note[1][6],
    // 81
    note[1][9], note[4][2], note[3][11], note[4][2], note[1][7], note[4][2], note[3][11], note[4][2],
    note[3][7], note[4][2], note[1][7], note[4][2], note[2][2], note[4][2], note[3][11], note[1][7],
    note[1][6], note[4][2], note[3][11], note[4][2], note[1][5], note[4][2], note[3][11], note[4][2],
    note[3][8], note[4][2], note[1][5], note[4][2], note[2][1], note[4][2], note[3][11], note[1][5],
    // 83
    note[1][6], note[4][1], note[1][6], note[4][1], note[1][9], note[4][1], note[1][6], note[4][1],
    note[1][2], note[4][6], note[1][2], note[4][6], note[1][6], note[4][6], note[1][2], note[4][6],
    note[1][0], note[4][6], note[4][3], note[4][6], note[3][6], note[4][6], note[4][3], note[4][6],
    note[3][6], note[4][6], note[1][0], note[4][6], note[1][3], note[4][6], note[1][0], note[4][6],
    // 85
    note[1][1], note[4][4], note[1][1], note[4][4], note[1][4], note[4][4], note[1][1], note[4][4],
    note[0][9], note[4][4], note[0][9], note[4][4], note[1][1], note[4][4], note[0][9], note[4][4],
    note[0][6], note[4][3], note[0][6], note[4][3], note[0][9], note[4][3], note[0][6], note[4][3],
    note[0][7], note[4][3], note[0][7], note[4][3], note[0][10], note[4][3], note[0][7], note[4][3],
    0
    }, {
    // 87
    note[4][3], note[1][8], note[0][8], note[1][8], note[3][8], note[1][8], note[0][8], note[1][8],
    note[0][8], note[1][8], note[0][8], note[1][8], note[3][9], note[1][8], note[3][8], note[1][8],
    note[3][6], note[1][8], note[0][8], note[1][8], note[3][8], note[1][8], note[3][6], note[1][8],
    note[3][4], note[1][8], note[0][8], note[1][8], note[3][6], note[1][8], note[3][4], note[1][8],
    note[3][3], note[1][8], note[0][8], note[1][8], note[3][4], note[1][8], note[3][3], note[1][8],
    note[3][1], note[1][8], note[0][8], note[1][8], note[3][3], note[1][8], note[3][1], note[1][8],
    // 90
    note[3][0], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8],
    note[3][1], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8],
    note[3][3], note[1][8], note[0][8], note[1][8], note[4][8], note[1][8], note[0][8], note[1][8],
    note[0][8], note[1][8], note[4][8], note[1][8], note[4][9], note[1][8], note[4][8], note[1][8],
    // 92
    note[4][6], note[1][8], note[4][6], note[1][8], note[4][8], note[1][8], note[4][6], note[1][8],
    note[4][4], note[1][8], note[4][4], note[1][8], note[4][6], note[1][8], note[4][4], note[1][8],
    note[4][3], note[1][8], note[4][3], note[1][8], note[4][4], note[1][8], note[4][3], note[1][8],
    note[4][1], note[1][8], note[4][1], note[1][8], note[4][3], note[1][8], note[4][1], note[1][8],
    // 94
    note[4][0], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8],
    note[4][1], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8],
    note[4][3], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8],
    note[0][8], note[1][8], note[0][8], note[1][8], note[4][1], note[1][8], note[0][8], note[1][8],
    // 96
    note[4][0], note[1][8], note[0][8], note[1][8], note[4][0], note[1][8], note[0][8], note[1][8],
    note[4][1], note[1][8], note[0][8], note[1][8], note[4][1], note[1][8], note[0][8], note[1][8],
    note[4][3], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8], note[0][8], note[1][8],
    note[0][8], note[1][8], note[0][8], note[1][8], note[3][4], note[1][8], note[4][1], note[1][8],
    // 98
    note[3][6], note[1][8], note[4][0], note[1][8], note[3][6], note[1][8], note[4][0], note[1][8],
    note[3][4], note[1][8], note[4][1], note[1][8], note[3][4], note[1][8], note[4][1], note[1][8],
    note[3][6], note[1][8], note[4][3], note[1][8], note[3][6], note[1][8], note[4][3], note[1][8],
    note[3][4], note[1][8], note[4][1], note[1][8], note[3][4], note[1][8], note[4][1], note[1][8],
    // 100
    note[3][9], note[3][8]
    }
};

extern int noteLength[song_number][song_length] = {{
    // 1
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 3
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 5
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 7
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 9
    2, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 11
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 13
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    4, 12,
    // 15
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 17
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 19
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0
    }, {
    // 21
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 23
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 25
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 27
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 29
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 31
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 33
    9,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 35
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 37
    1, 3, 5,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 39
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 41
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0
    }, {
    // 43
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 45
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 47
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 49
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 51
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 53
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 55
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 57
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 59
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 61
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 63
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0
    }, {
    // 1
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 3
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 5
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 7
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 9
    2, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 11
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 13
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    4, 12,
    // 15
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 17
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 19
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0
    }, {
    // 21
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 23
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 25
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 27
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 29
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 31
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 33
    9,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 35
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 37
    1, 3, 5,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 39
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 41
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0
    }, {
    // 43
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 45
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 47
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 49
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 51
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 53
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 55
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    // 57
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 59
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 61
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 63
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0
    }, {
    // 65
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 67
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 69
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2,
    // 71
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 73
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 75
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 77
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 79
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 81
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 83
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 85
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0
    }, {
    // 87
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 90
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 92
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 94
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 96
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 98
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    // 100
    16, 16,
    0
    }
};

extern int Taiko_song[Taiko_length] = {0};
extern int Taiko_noteLength[Taiko_length] = {0};
extern int Taiko_beatNote[Taiko_length] = {0};

#endif