import { FileText, Database, Network, TrendingUp } from "lucide-react";
import { Card } from "@/components/ui/card";

interface StatsCardsProps {
  documentsCount: number;
  factsCount: number;
  nodesCount: number;
  connectionsCount: number;
}

export function StatsCards({
  documentsCount,
  factsCount,
  nodesCount,
  connectionsCount,
}: StatsCardsProps) {
  const stats = [
    {
      label: "Documents",
      value: documentsCount,
      icon: FileText,
      color: "text-chart-1",
    },
    {
      label: "Facts Extracted",
      value: factsCount,
      icon: Database,
      color: "text-chart-2",
    },
    {
      label: "Graph Nodes",
      value: nodesCount,
      icon: Network,
      color: "text-chart-3",
    },
    {
      label: "Connections",
      value: connectionsCount,
      icon: TrendingUp,
      color: "text-chart-4",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <Card key={stat.label} className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground mb-1">{stat.label}</p>
              <p className="text-3xl font-semibold" data-testid={`stat-${stat.label.toLowerCase().replace(' ', '-')}`}>
                {stat.value}
              </p>
            </div>
            <stat.icon className={`h-8 w-8 ${stat.color}`} />
          </div>
        </Card>
      ))}
    </div>
  );
}
