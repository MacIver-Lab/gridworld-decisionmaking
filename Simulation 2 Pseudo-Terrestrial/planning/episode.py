import csv

class ENTRY:
    def __init__(self, action, observation, reward, state):
        self.Action = action
        self.Observation = observation
        self.Reward = reward
        self.State = state

class Episode:
    def __init__(self):
        self.EpisodeVector = []
        self.AllEpisodes = []

    def Add(self, action, observation, state, reward=0):
        self.EpisodeVector.append(ENTRY(action, observation, reward, state))

    def Complete(self):
        self.AllEpisodes.append(self.EpisodeVector)

    def Pop(self):
        self.EpisodeVector = self.EpisodeVector[:-1]

    def Truncate(self, t):
        self.EpisodeVector = self.EpisodeVector[:t]

    def Forget(self, t):
        self.EpisodeVector = self.EpisodeVector[t:]

    def Clear(self):
        self.EpisodeVector = []

    def ClearAll(self):
        self.AllEpisodes = []
        self.EpisodeVector = []

    def Size(self):
        return len(self.EpisodeVector)

    def Back(self):
        assert(self.Size() > 0)
        return self.EpisodeVector[-1]

    def __eq__(self, other):
        if other.Size() != self.Size():
            return False

        for i, episode in enumerate(other):
            if episode.BeliefState != self.EpisodeVector[i].BeliefState or \
                episode.Action != self.EpisodeVector[i].Action or \
                episode.Observation != self.EpisodeVector[i].Observation:

                return False

        return True

    def Episode2CSV(self, filename):
        episodeDict = {}
        episodeDict['Action'] = []
        episodeDict['Observation'] = []
        episodeDict['Predator X'] = []
        episodeDict['Predator Y'] = []
        episodeDict['Agent X'] = []
        episodeDict['Agent Y'] = []
        episodeDict['Reward'] = []

        for episode in self.EpisodeVector:
            episodeDict['Action'].append(episode.Action)
            episodeDict['Predator X'].append(episode.State.PredatorPos.X)
            episodeDict['Predator Y'].append(episode.State.PredatorPos.Y)
            episodeDict['Agent X'].append(episode.State.AgentPos.X)
            episodeDict['Agent Y'].append(episode.State.AgentPos.Y)
            episodeDict['Observation'].append(episode.Observation)
            episodeDict['Reward'].append(episode.Reward)

        columns = sorted(episodeDict)
        with open(filename, 'w') as f:
            writer = csv.writer(f); writer.writerow(columns); writer.writerows(zip(*[episodeDict[col] for col in columns]))

    def __getitem__(self, t):
        assert(t>=0 and t<self.Size())
        return self.EpisodeVector[t]

    def Display(self):
        for elem in self.EpisodeVector:
            print("state: ", elem.State)
            print("action: ", elem.Action)
            print("observation: ", elem.Observation)
            print("reward: ", elem.Reward)