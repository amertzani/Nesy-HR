import { useQuery } from "@tanstack/react-query";
import { hfApi } from "@/lib/api-client";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Database, Network, Bot, AlertCircle, BookOpen, Link2, Tag } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function OntologyPage() {
  const { data: ontologyData, isLoading, error } = useQuery({
    queryKey: ["ontology"],
    queryFn: async () => {
      const response = await hfApi.getOntology();
      if (response.success) {
        return response.data?.ontology;
      }
      throw new Error(response.error || "Failed to load ontology");
    },
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <AlertCircle className="h-8 w-8 text-destructive" />
        <p className="text-muted-foreground">Failed to load ontology</p>
        <p className="text-sm text-muted-foreground">{error instanceof Error ? error.message : String(error)}</p>
      </div>
    );
  }

  if (!ontologyData || (typeof ontologyData === 'object' && Object.keys(ontologyData).length === 0)) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <Database className="h-12 w-12 text-muted-foreground" />
        <p className="text-muted-foreground">Ontology not initialized yet</p>
        <p className="text-sm text-muted-foreground">Upload a document to initialize the ontology</p>
      </div>
    );
  }

  const entities = ontologyData.entities || [];
  const relationships = ontologyData.relationships || [];
  const properties = ontologyData.properties || {};
  const domain = ontologyData.domain || "Unknown Domain";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Ontology Schema</h1>
        <p className="text-muted-foreground">
          Comprehensive view of the knowledge graph ontology for {domain}
        </p>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="entities">Entities ({entities.length})</TabsTrigger>
          <TabsTrigger value="relationships">Relationships ({relationships.length})</TabsTrigger>
          <TabsTrigger value="properties">Properties</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <Database className="h-8 w-8 text-primary" />
                </div>
                <div className="text-3xl font-bold">{entities.length}</div>
                <div className="text-sm text-muted-foreground">Entity Types</div>
              </div>
              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <Link2 className="h-8 w-8 text-primary" />
                </div>
                <div className="text-3xl font-bold">{relationships.length}</div>
                <div className="text-sm text-muted-foreground">Relationships</div>
              </div>
              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <Tag className="h-8 w-8 text-primary" />
                </div>
                <div className="text-3xl font-bold">
                  {Object.values(properties).reduce((sum: number, props: any) => sum + (props?.length || 0), 0)}
                </div>
                <div className="text-sm text-muted-foreground">Total Properties</div>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BookOpen className="h-5 w-5" />
              Domain Information
            </h3>
            <div className="space-y-2">
              <div>
                <span className="font-medium">Domain:</span>{" "}
                <Badge variant="outline">{domain}</Badge>
              </div>
              <div>
                <span className="font-medium">Status:</span>{" "}
                <Badge className="bg-green-500 text-white">
                  {ontologyData.ontology_initialized !== false ? "Initialized" : "Not Initialized"}
                </Badge>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Quick Stats</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-muted-foreground">Most Properties</div>
                <div className="text-lg font-semibold">
                  {Object.entries(properties).reduce((max, [entity, props]) => 
                    (props?.length || 0) > (max[1]?.length || 0) ? [entity, props] : max, 
                    ["", []] as [string, any[]]
                  )[0] || "N/A"}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Average Properties per Entity</div>
                <div className="text-lg font-semibold">
                  {entities.length > 0
                    ? (Object.values(properties).reduce((sum: number, props: any) => sum + (props?.length || 0), 0) / entities.length).toFixed(1)
                    : "0"}
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="entities" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {entities.map((entity: string) => (
              <Card key={entity} className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Database className="h-5 w-5 text-blue-500" />
                    <h3 className="font-semibold">{entity}</h3>
                  </div>
                  <Badge variant="outline">
                    {properties[entity]?.length || 0} props
                  </Badge>
                </div>
                {properties[entity] && properties[entity].length > 0 && (
                  <div className="mt-3 space-y-1">
                    <div className="text-xs font-medium text-muted-foreground">Properties:</div>
                    <div className="flex flex-wrap gap-1">
                      {properties[entity].map((prop: string) => (
                        <Badge key={prop} variant="secondary" className="text-xs">
                          {prop}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="relationships" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {relationships.map((rel: string) => (
              <Card key={rel} className="p-4">
                <div className="flex items-center gap-2">
                  <Link2 className="h-5 w-5 text-purple-500" />
                  <h3 className="font-semibold">{rel}</h3>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  Relationship type
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="properties" className="space-y-4">
          {Object.entries(properties).map(([entity, props]: [string, any]) => (
            <Card key={entity} className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <Database className="h-5 w-5 text-blue-500" />
                <h3 className="font-semibold text-lg">{entity}</h3>
                <Badge variant="outline">{props?.length || 0} properties</Badge>
              </div>
              {props && props.length > 0 ? (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                  {props.map((prop: string) => (
                    <div
                      key={prop}
                      className="p-2 rounded-md bg-muted text-sm flex items-center gap-2"
                    >
                      <Tag className="h-3 w-3 text-muted-foreground" />
                      {prop}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No properties defined</p>
              )}
            </Card>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  );
}

